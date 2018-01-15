use crossbeam_epoch::*;
use std::hash::{Hash, BuildHasher};
use std::sync::atomic::Ordering;
use std::borrow::Borrow;
use std::mem;
use std::ptr;
use std::fmt;

/// A lock-free concurrent skiplist-based map with epoch GC memory reclamation.
///
/// This structure is based on Keir Fraser's algorithm from [his PhD thesis]
/// (https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-579.pdf).
// TODO make Debug not rely on K: Debug, V: Debug, S: Debug.
#[derive(Debug)]
pub struct EpochSkiplistMap<K, V, S> {
    head: Atomic<Node<K, V>>,
    hasher_factory: S,
}

// TODO i just guessed these.
unsafe impl<K, V, S> Send for EpochSkiplistMap<K, V, S>
    where K: Send,
          V: Send,
          S: Send {}
unsafe impl<K, V, S> Sync for EpochSkiplistMap<K, V, S>
    where K: Send + Sync,
          V: Send + Sync,
          S: Sync {}

const HEIGHT: usize = 16;
const TOP_LEVEL: usize = HEIGHT - 1;

#[derive(Debug)]
enum Node<K, V> {
    Head {
        nexts: [Atomic<Node<K, V>>; HEIGHT],
    },
    Data {
        hash: u64,
        key: K,
        val: Option<V>,

        nexts: [Atomic<Node<K, V>>; HEIGHT],
        top_level: usize,
    },
    Tail,
}

impl<K, V> Node<K, V> {
    fn nexts(&self) -> &[Atomic<Node<K, V>>; HEIGHT] {
        use self::Node::*;

        match *self {
            Head { ref nexts } | Data { ref nexts, .. } => nexts,
            _ => panic!("!!"),
        }
    }

    fn is_data(&self) -> bool {
        use self::Node::*;

        if let Data { .. } = *self { true } else { false }
    }

    fn is_tail(&self) -> bool {
        use self::Node::*;

        if let Tail = *self { true } else { false }
    }

    fn key(&self) -> &K {
        use self::Node::*;

        if let Data { ref key, .. } = *self { key } else { panic!("!!") }
    }

    fn val(&self) -> &V {
        use self::Node::*;

        if let Data { ref val, .. } = *self { val.as_ref().unwrap() } else { panic!("!!") }
    }

    fn val_opt_mut(&mut self) -> &mut Option<V> {
        use self::Node::*;

        if let Data { ref mut val, .. } = *self { val } else { panic!("!!") }
    }

    fn hash(&self) -> u64 {
        use self::Node::*;

        if let Data { hash, .. } = *self { hash } else { panic!("!!") }
    }

    fn top_level(&self) -> usize {
        use self::Node::*;

        match *self {
            Data { top_level, .. } => top_level,
            Head { .. } | Tail => TOP_LEVEL,
        }
    }
}

impl<K, V, S> Drop for EpochSkiplistMap<K, V, S> {
    fn drop(&mut self) {
        // We pin three times to GC any leftovers. This may not be necessary, but it doesn't hurt.
        let _ = pin();
        let _ = pin();
        let guard = pin();
        let mut pred: Shared<Node<K, V>> = self.head.load(Ordering::Relaxed, &guard);

        unsafe {
            while !pred.deref().is_tail() {
                let curr: Shared<Node<K, V>> = pred.deref().nexts()[0].load(Ordering::Relaxed, &guard);
                // Drop is statically guaranteed to only run on a single thread - we can deallocate 'by hand'.
                Box::from_raw(pred.as_raw() as *mut Node<K, V>);
                pred = curr;
            }

            // Dispose of tail.
            Box::from_raw(pred.as_raw() as *mut Node<K, V>);
        }
    }
}

struct NodePosition<'a, K: 'a, V: 'a> {
    preds: [Shared<'a, Node<K, V>>; HEIGHT],
    currs_or_nexts: [Shared<'a, Node<K, V>>; HEIGHT],
}

impl<'a, K: 'a, V: 'a> fmt::Debug  for NodePosition<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NodePosition")
            .finish()
    }
}


impl<K, V, S> EpochSkiplistMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    /// Creates an empty map which will use the given hasher factory to hash keys.
    pub fn with_hash_factory(f: S) -> Self {
        use init_with::InitWith;

        let tail = Box::into_raw(Box::new(Node::Tail));
        EpochSkiplistMap {
            head: Atomic::new(
                Node::Head {
                    nexts: <[Atomic<Node<K, V>>; HEIGHT]>::init_with(|| {
                        unsafe { Atomic::from(Owned::from_raw(tail)) }
                    } ),
                }
            ),
            hasher_factory: f,
        }
    }

    /// Hashes the given key using the hasher factory which this map instance uses.
    fn hash<Q: ?Sized + Hash>(&self, key: &Q) -> u64 {
        use std::hash::Hasher;

        // TODO is this stateless?
        let mut hasher = self.hasher_factory.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// If a node with the given key is found, returns `Ok` containing its predecessor and either itself for each level
    /// where it's linked or its sucessor for each level where it's not linked. If a node with the given key is not found,
    /// returns `Err` containing the first node whose hash is larger than the given key's hash and its predecessor at every
    /// level. That predecessor may not have the same key as the given key, but may have the same hash. This method also
    /// unlinks all marked nodes it encounters.
    fn find_pairs<'a, Q: ?Sized>(&self, key: &Q, g: &'a Guard) -> Result<NodePosition<'a, K, V>, NodePosition<'a, K, V>>
        where K: Borrow<Q>,
              Q: Eq + Hash
    {
        let hash = self.hash(key);

        // "The Position" refers to the position where a node with the given key should be or is.
        // This list points to nodes before the Position at every level.
        // TODO what about panics? might be UB, are there other UB situations possible?
        let mut preds: [Shared<Node<K, V>>; HEIGHT] = unsafe { mem::uninitialized() };
        // This list at every level points to nodes either after the Position or at the Position if a node with the given key exists.
        let mut currs: [Shared<Node<K, V>>; HEIGHT] = unsafe { mem::uninitialized() };

        unsafe {
            // We execute the standard skiplist search algorithm by moving right and descending levels when the next node comes after the Position.
            'begin_from_head: loop {
                let mut pred: Shared<Node<K, V>> = self.head.load(Ordering::SeqCst, g);
                for lvl in (0..=TOP_LEVEL).rev() {
                    let mut pred_next: Shared<Node<K, V>> = pred.deref().nexts()[lvl].load(Ordering::SeqCst, g);
                    if pred_next.tag() == 1 {
                        // TODO why? Fraser does this, H&S doesn't.
                        continue 'begin_from_head;
                    }

                    let mut curr: Shared<Node<K, V>> = pred_next;
                    // Loop until we encounter Node::Tail
                    while curr.deref().is_data() {
                        let mut curr_next: Shared<Node<K, V>> = curr.deref().nexts()[lvl].load(Ordering::SeqCst, g);
                        // Skip over a sequence of marked nodes if such exists
                        while curr.deref().is_data() {
                            curr_next = curr.deref().nexts()[lvl].load(Ordering::SeqCst, g);
                            if curr_next.tag() != 1 {
                                break;
                            }
                            // If this node is marked, we keep pred the same and only increment curr so that the CAS below can remove it.
                            curr = curr_next;
                        }

                        if !curr.deref().is_data() {
                            break;
                        }

                        if (curr.deref().hash() == hash && curr.deref().key().borrow() == key) || curr.deref().hash() > hash {
                            // The next node falls after the Position, we're done on this level.
                            break;
                        }

                        pred = curr;
                        pred_next = curr_next;
                        curr = curr_next;
                    }

                    // Pred and curr can be not adjacent if there is a sequence of marked nodes between them
                    // or if a new node was inserted between them during our search.
                    if (pred_next.as_raw() != curr.as_raw())
                        // TODO this probably only removes marked nodes if they directly precede the Position. Not strictly incorrect, but might hog memory.
                        // This CAS only fires for the case of marked nodes, since removing an inserted one would be incorrect.
                        && pred.deref().nexts()[lvl].compare_and_set(
                            pred_next,
                            curr.with_tag(pred_next.tag()),
                            Ordering::SeqCst,
                            &g).is_err()
                    {
                        // If we fail, something changed in the neighbourhood, so restart search.
                        continue 'begin_from_head;
                    }

                    ptr::write(&mut preds[lvl] as *mut _, pred);
                    ptr::write(&mut currs[lvl] as *mut _, curr);
                }

                if currs[0].deref().is_data() && currs[0].deref().key().borrow() == key {
                    return Ok(NodePosition {
                        preds: preds,
                        // currs
                        currs_or_nexts: currs,
                    } );
                }
                else {
                    return Err(NodePosition {
                        preds: preds,
                        // nexts
                        currs_or_nexts: currs,
                    } );
                }
            }
        }
    }

    /// Returns a randomly chosen level in range [0;HEIGHT), where the probability
    /// of choosing level L is PROB^(L+1).
    // PROB is currently hardwired to 1/2.
    fn random_level(&self) -> usize {
        use rand;

        // thanks to http://ticki.github.io/blog/skip-lists-done-right/
        // TODO think about what rng to use, thread-local or not, etc
        let r: u64 = rand::random::<u64>() & ((1 << HEIGHT) - 1);
        for i in 0..HEIGHT {
            if (r >> i) & 1 == 1 {
                return i;
            }
        }
        HEIGHT - 1
    }

    /// Inserts a key-value pair into the map. Fails if a node with the given key already exists. Returns whether
    /// the insert was successful.
    pub fn insert(&self, key: K, val: V) -> bool {
        use init_with::InitWith;

        let guard = pin();
        let top_level = self.random_level();
        let mut new = Owned::new(Node::Data {
            hash: self.hash(&key),
            key: key,
            val: Some(val),
            nexts: <[Atomic<Node<K, V>>; HEIGHT]>::init_with(|| {
                Atomic::null()
            } ),
            top_level: top_level,
        } ).into_shared(&guard);

        unsafe {
            loop {
                match self.find_pairs(new.deref().key(), &guard) {
                    Ok(_) => {
                        // Key is already used, operation failed.
                        return false;
                    },
                    Err(acc) => {
                        let mut acc = acc;

                        // Set the new node's nexts to point to locations following where it should be.
                        for lvl in (0..=new.deref().top_level()).rev() {
                            // TODO with_tag here.. wot?
                            new.deref().nexts()[lvl].store(acc.currs_or_nexts[lvl].with_tag(0), Ordering::SeqCst);
                        }

                        // Inserting the node into the bottom level logically adds it to the map.
                        let new_ref = match acc.preds[0].deref().nexts()[0].compare_and_set(
                            acc.currs_or_nexts[0].with_tag(0),
                            new.with_tag(0),
                            Ordering::SeqCst,
                            &guard)
                        {
                            Ok(new) => {
                                new
                            },
                            Err(cas_err) => {
                                new = cas_err.new;
                                // We failed to insert the node, retry.
                                continue;
                            },
                        };

                        // Inserting the node into higher levels must be done with care,
                        // since it's already accessible from the list and must be valid at all times.
                        for lvl in 1..=top_level {
                            loop {
                                //let pred = acc.preds[lvl];
                                let /*mut*/ next = acc.currs_or_nexts[lvl];

                                // We have to check whether the new node's forward pointers are still valid at each level before
                                // inserting it there.
                                let new_next = new_ref.deref().nexts()[lvl].load(Ordering::SeqCst, &guard);
                                if (new_next.as_raw() != next.as_raw())
                                    && new_ref.deref().nexts()[lvl].compare_and_set(new_next.with_tag(0), next.with_tag(0), Ordering::SeqCst, &guard).is_err() {
                                        break; // Give up if pointer is marked
                                    }

                                // If the list is already inserted at this level.. wat?
                                // TODO Fraser does this, but it's impossible for a node to already be at a level where it
                                // wasn't yet inserted.
                                /*if next.is_data() && next.key() == new_ref.key() {
                                next = next.nexts()[lvl].load(Ordering::SeqCst, &guard).0.unwrap();
                            }*/

                                // Insert the node at the given level.
                                if acc.preds[lvl].deref().nexts()[lvl].compare_and_set(next.with_tag(0), new_ref.with_tag(0), Ordering::SeqCst, &guard).is_ok() {
                                    break;
                                }

                                // If we failed to insert above, redo search to find out where our node should be.
                                acc = match self.find_pairs(new_ref.deref().key(), &guard) {
                                    // TODO doesn't Err here mean that we already got deleted and should abandon linking higher levels?
                                    Ok(acc) | Err(acc) => acc,
                                };
                            }
                        }

                        return true;
                    },
                }
            }
        }
    }

    /// If a node with the given key is found, removes it from the map and returns its value in `Some`.
    /// Otherwise returns `None`.
    pub fn remove<Q: ?Sized>(&self, key: &Q) -> Option<V>
        where K: Borrow<Q>,
              Q: Eq + Hash
    {
        let guard = pin();

        unsafe {
            // Maps Err to None, since if a node wasn't found there is nothing to remove
            // and then executes remove on Ok(acc: NodePosition).
            self.find_pairs(key, &guard).ok().and_then(|acc| {
                let node_to_remove: Shared<Node<K, V>> = acc.currs_or_nexts[0];
                // First mark all levels besides 0 top-to-bottom. Node is still logically in the map.
                for lvl in (1..=node_to_remove.deref().top_level()).rev() {
                    let mut node_next = node_to_remove.deref().nexts()[lvl].load(Ordering::SeqCst, &guard);
                    while node_next.tag() != 1 {
                        node_to_remove.deref().nexts()[lvl].compare_and_set(node_next.with_tag(0), node_next.with_tag(1), Ordering::SeqCst, &guard).is_ok();
                        node_next = node_to_remove.deref().nexts()[lvl].load(Ordering::SeqCst, &guard);
                    }
                }

                let mut node_next = node_to_remove.deref().nexts()[0].load(Ordering::SeqCst, &guard);
                let mut i_marked_it = false;
                while node_next.tag() != 1 {
                    // Marking the bottom level reference logically removes the node from the map. The thread which succeeds
                    // in doing so must also move the value out.
                    i_marked_it = node_to_remove.deref().nexts()[0].compare_and_set(node_next.with_tag(0), node_next.with_tag(1), Ordering::SeqCst, &guard).is_ok();
                    node_next = node_to_remove.deref().nexts()[0].load(Ordering::SeqCst, &guard);
                }

                if i_marked_it {
                    // TODO optimize and try to CAS acc.preds instead.
                    self.find_pairs(key, &guard).is_ok();

                    // This is fine since find_pairs is guaranteed to unlink (make unreachable) the node we search for.
                    let mut owned = node_to_remove.into_owned();
                    let val = owned.val_opt_mut().take().unwrap();

                    guard.defer(move || drop(owned));

                    Some(val)
                } else {
                    None
                }
            } )
        }
    }

    /// If a node with the given key exists, returns a shared pointer to it in `Some`.
    /// Otherwise returns `None`. Unlike `find_pairs`, this operation does not unlink marked
    /// nodes and hence is faster.
    fn find_no_cleanup<'a, Q: ?Sized>(&self, key: &Q, g: &'a Guard) -> Option<Shared<'a, Node<K, V>>>
        where K: Borrow<Q>,
              Q: Eq + Hash
    {
        let hash = self.hash(key);

        unsafe {
            let mut pred: Shared<Node<K, V>> = self.head.load(Ordering::SeqCst, g);
            for lvl in (0..=TOP_LEVEL).rev() {
                let mut curr: Shared<Node<K, V>> = pred.deref().nexts()[lvl].load(Ordering::SeqCst, g);
                while curr.deref().is_data() {
                    let curr_next: Shared<Node<K, V>> = curr.deref().nexts()[lvl].load(Ordering::SeqCst, g);

                    if curr_next.tag() != 1 && curr.deref().hash() == hash && curr.deref().key().borrow() == key {
                        return Some(curr);
                    } else if curr.deref().hash() > hash {
                        break;
                    }

                    pred = curr;
                    curr = curr_next;
                }
            }
        }

        None
    }

    /// Returns whether the map contains a node with the given key.
    pub fn contains<Q: ?Sized>(&self, key: &Q) -> bool
        where K: Borrow<Q>,
              Q: Eq + Hash
    {
        let guard = pin();
        self.find_no_cleanup(key, &guard).is_some()
    }

    /// If a node with the given key exists, returns the clone of its value in `Some`.
    /// Otherwise returns `None`.
    pub fn find<Q: ?Sized>(&self, key: &Q) -> Option<V>
        where K: Borrow<Q>,
              Q: Eq + Hash,
              V: Clone
    {
        let guard = pin();
        self.find_no_cleanup(key, &guard).and_then(|node| {
            unsafe {
                Some(node.deref().val().clone())
            }
        } )
    }

    /// Returns the size of the map, that is the amount of key-value pairs in it.
    pub fn size(&self) -> usize {
        let guard = pin();
        let mut count = 0;

        unsafe {
            let mut curr: Shared<Node<K, V>> = self.head.load(Ordering::SeqCst, &guard).deref().nexts()[0].load(Ordering::SeqCst, &guard);
            while curr.deref().is_data() {
                let pr = curr.deref().nexts()[0].load(Ordering::SeqCst, &guard);
                if pr.tag() != 1 {
                    count += 1;
                }
                curr = pr;
            }
        }

        count
    }

    fn _swap(&self, _key: &K, _new: V) -> Option<V> {
        // TODO? Which atomic operations do we want to support?
        None
    }
}

#[cfg(test)]
mod tests {
    use std::hash::BuildHasherDefault;
    use std::collections::hash_map::DefaultHasher;

    use std::sync::Barrier;

    use crossbeam_utils::scoped::scope;

    use super::*;

    fn with_def_hasher<K, V>() -> EpochSkiplistMap<K, V, BuildHasherDefault<DefaultHasher>>
        where K: Eq + Hash
    {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        EpochSkiplistMap::with_hash_factory(build)
    }

    fn st_insert_k(k_loops: usize) {
        let map: EpochSkiplistMap<usize, usize, _> = with_def_hasher();
        for i in 0..k_loops {
            let inserted = map.insert(i, i+1);
            assert!(inserted);
            assert!(map.contains(&i));
            assert!(map.find(&i) == Some(i+1));
            assert!(!map.contains(&(i+1)));
        }
    }

    #[test]
    fn st_insert() {
        st_insert_k(1);
        st_insert_k(10);
    }

    fn st_remove_k(k_loops: usize) {
        let map: EpochSkiplistMap<usize, usize, _> = with_def_hasher();
        for i in 0..k_loops {
            map.insert(i, i+1);
        }

        for i in 0..k_loops {
            let removed = map.remove(&i);
            assert!(removed.is_some());
            assert!(removed.unwrap() == i+1);
            assert!(!map.contains(&i));
            let removed_twice = map.remove(&i);
            assert!(removed_twice.is_none());
        }
    }

    #[test]
    fn st_remove() {
        st_remove_k(1);
        st_remove_k(10);
    }

    #[test]
    fn st_size() {
        let map: EpochSkiplistMap<usize, usize, _> = with_def_hasher();
        for i in 0..10 {
            assert!(map.size() == i);
            map.insert(i, i+1);
        }
    }

    fn st_leak_k(k_loops: usize) {
        let map: EpochSkiplistMap<Box<usize>, Box<usize>, _> = with_def_hasher();

        for i in 0..k_loops {
            map.insert(Box::new(i), Box::new(i));
        }
        for i in 0..(k_loops/2) {
            map.remove(&Box::new(i));
        }
    }

    #[test]
    fn st_leak() {
        st_leak_k(1);
        st_leak_k(10);
    }

    struct U32(u32);
    fn mt_n_test<F: Fn(usize)>(f: F) {
        f(2);
        f(4);
        f(8);
    }

    fn mt_n_insert_k(n_threads: usize, k_loops: usize) {
        let map: EpochSkiplistMap<u32, u32, _> = with_def_hasher();
        let b = Barrier::new(n_threads);
        scope(|scope| {
            for i in 0..n_threads {
                let i = U32(i as u32);
                scope.spawn(|| {
                    let i = {i}.0;
                    b.wait();
                    for j in 0..k_loops {
                        let i = (k_loops as u32)*i + j as u32;
                        let inserted = map.insert(i, i+1);
                        assert!(inserted);
                        assert!(map.contains(&i));
                        assert!(map.find(&i) == Some(i+1));
                    }
                } );
            }
        } );
    }

    #[test]
    fn mt_insert() {
        mt_n_test(|n| { mt_n_insert_k(n, 1) } );
        mt_n_test(|n| { mt_n_insert_k(n, 100) } );
    }

    fn mt_n_remove_k(n_threads: usize, k_loops: usize) {
        let map: EpochSkiplistMap<u32, u32, _> = with_def_hasher();
        let b = Barrier::new(n_threads);
        scope(|scope| {
            for i in 0..n_threads {
                let i = U32(i as u32);
                scope.spawn(|| {
                    let i = {i}.0;
                    b.wait();
                    for j in 0..k_loops {
                        let i = k_loops as u32*i + j as u32;
                        map.insert(i, i+1);
                    }

                    for j in 0..k_loops {
                        let i = k_loops as u32*i + j as u32;
                        let removed = map.remove(&i);
                        assert!(removed.is_some());
                        assert!(removed.unwrap() == i+1);
                        assert!(!map.contains(&i));
                        let removed_twice = map.remove(&i);
                        assert!(removed_twice.is_none());
                    }
                } );
            }
        } );
    }

    #[test]
    fn mt_remove() {
        mt_n_test(|n| { mt_n_remove_k(n, 1) } );
        mt_n_test(|n| { mt_n_remove_k(n, 100) } );
    }

    fn mt_n_afl(_n_threads: usize) {
        // TODO
    }

    #[test]
    fn mt_afl() {
        mt_n_test(mt_n_afl);
    }
}
