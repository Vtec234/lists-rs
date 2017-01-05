use crossbeam::mem::epoch::*;
use std::hash::{Hash, BuildHasher};
use std::sync::atomic::Ordering;
use std::mem;
use std::ptr;

pub struct EpochSkiplistMap<K, V, S> {
    head: Atomic<Node<K, V>>,
    hasher_factory: S,
}

const HEIGHT: usize = 16;
const TOP_LEVEL: usize = HEIGHT - 1;

#[derive(Debug)]
enum Node<K, V> {
    Head {
        nexts: [MarkableAtomic<Node<K, V>>; HEIGHT],
    },
    Data {
        hash: u64,
        key: K,
        val: V,

        nexts: [MarkableAtomic<Node<K, V>>; HEIGHT],
        top_level: usize,
    },
    Tail,
}

impl<K, V> Node<K, V> {
    fn nexts(&self) -> &[MarkableAtomic<Node<K, V>>; HEIGHT] {
        use self::Node::*;

        match self {
            &Head { ref nexts } => nexts,
            &Data { ref nexts, .. } => nexts,
            _ => panic!("1"),
        }
    }

    fn is_data(&self) -> bool {
        use self::Node::*;

        if let &Data { .. } = self { true } else { false }
    }

    fn key(&self) -> &K {
        use self::Node::*;

        if let &Data { ref key, .. } = self { key } else { panic!("2") }
    }

    fn val(&self) -> &V {
        use self::Node::*;

        if let &Data { ref val, .. } = self { val } else { panic!("20") }
    }

    fn hash(&self) -> u64 {
        use self::Node::*;

        if let &Data { hash, .. } = self { hash } else { panic!("3") }
    }

    fn top_level(&self) -> usize {
        use self::Node::*;

        match self {
            &Data { top_level, .. } => top_level,
            &Head { .. } => TOP_LEVEL,
            &Tail => TOP_LEVEL,
        }
    }

    unsafe fn val_cpy(&self) -> V {
        use self::Node::*;

        if let &Data { ref val, .. } = self { ptr::read(val as *const _ as *mut _) } else { panic!("21") }
    }
}

impl<K, V, S> Drop for EpochSkiplistMap<K, V, S> {
    fn drop(&mut self) {
        println!("drop");
    }
}

struct NodePosition<'a, K: 'a, V: 'a> {
    preds: [Shared<'a, Node<K, V>>; HEIGHT],
    currs_or_nexts: [Shared<'a, Node<K, V>>; HEIGHT],
}

use std::fmt;
impl<K, V, S> EpochSkiplistMap<K, V, S> where K: Eq + Hash, S: BuildHasher {
    pub fn with_hash_factory(f: S) -> Self {
        use init_with::InitWith;

        let tail = Box::into_raw(Box::new(Node::Tail));
        EpochSkiplistMap {
            head: Atomic::new(
                Node::Head {
                    nexts: <[MarkableAtomic<Node<K, V>>; HEIGHT]>::init_with(|| {
                        unsafe { MarkableAtomic::from_ptr(tail, false) }
                    } ),
                }
            ),
            hasher_factory: f,
        }
    }

    fn hash(&self, key: &K) -> u64 {
        use std::hash::Hasher;

        // TODO is this stateless?
        let mut hasher = self.hasher_factory.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
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
        return HEIGHT - 1;
    }

    fn find_pairs<'a>(&self, key: &K, g: &'a Guard) -> Result<NodePosition<'a, K, V>, NodePosition<'a, K, V>> {
        use init_with::InitWith;

        let hash = self.hash(key);

        // "The Position" refers to the position where a node with the given key should be or is.
        // This list points to nodes before the Position at every level.
        // TODO this shouldn't be Option, it's annoying. Do ptr::write? We can probably count on Shared's destructor not to do anything wierd with junk memory.
        let mut preds = <[Option<Shared<Node<K, V>>>; HEIGHT]>::init_with(|| {
            None
        } );
        // This list at every level points to nodes either after the Position or at the Position if a node with the given key exists.
        let mut currs = <[Option<Shared<Node<K, V>>>; HEIGHT]>::init_with(|| {
            None
        } );

        // We execute the standard skiplist search algorithm by moving right and descending levels when the next node comes after the Position.
        'begin_from_head: loop {
            let mut pred: Shared<Node<K, V>> = self.head.load(Ordering::SeqCst, g).unwrap();
            for lvl in (0...TOP_LEVEL).rev() {
                let mut pred_next: (Option<Shared<Node<K, V>>>, bool) = pred.nexts()[lvl].load(Ordering::SeqCst, g);
                if pred_next.1 {
                    // TODO why? Fraser does this, H&S doesn't.
                    continue 'begin_from_head;
                }

                let mut curr: Shared<Node<K, V>> = pred_next.0.unwrap();
                // Loop until we encounter Node::Tail
                while curr.is_data() {
                    let mut curr_next: (Option<Shared<Node<K, V>>>, bool) = curr.nexts()[lvl].load(Ordering::SeqCst, g);
                    // Skip over a sequence of marked nodes if such exists
                    while curr.is_data() {
                        curr_next = curr.nexts()[lvl].load(Ordering::SeqCst, g);
                        if !curr_next.1 {
                            break;
                        }
                        // If this node is marked, we keep pred the same and only increment curr so that the CAS below can remove it.
                        curr = curr_next.0.unwrap();
                    }

                    if !curr.is_data() {
                        break;
                    }

                    if (curr.hash() == hash && curr.key() == key) || curr.hash() > hash {
                        // The next node falls after the Position, we're done on this level.
                        break;
                    }

                    pred = curr;
                    pred_next = curr_next;

                    curr = curr_next.0.unwrap();
                }
                // TODO what if TAIL should be curr? this will be wrong then

                // Pred and curr can be not adjacent if there is a sequence of marked nodes between them
                // or if a new node was inserted between them during our search.
                if (pred_next.0.unwrap().as_raw() != curr.as_raw())
                // TODO this probably only removes marked nodes if they directly precede the Position. Not strictly incorrect, but might hog memory.
                // This CAS only fires for the case of marked nodes, since removing an inserted one would be incorrect.
                && !pred.nexts()[lvl].cas_shared(
                    pred_next.0, pred_next.1,
                    Some(curr), pred_next.1,
                    Ordering::SeqCst)
                {
                    // If we fail, something changed in the neighbourhood, so restart search.
                    continue 'begin_from_head;
                }

                preds[lvl] = Some(pred);
                currs[lvl] = Some(curr);
            }

            // TODO make this less hacky by setting the same type, see definitions above.
            let mut i = 0;
            let mut j = 0;
            if currs[0].unwrap().is_data() && currs[0].unwrap().key() == key {
                return Ok(NodePosition {
                    preds: <[Shared<Node<K, V>>; HEIGHT]>::init_with(|| {
                        let k = i;
                        i = i + 1;
                        preds[k].unwrap()
                    } ),
                    currs_or_nexts: <[Shared<Node<K, V>>; HEIGHT]>::init_with(|| {
                        let k = j;
                        j = j + 1;
                        currs[k].unwrap()
                    } ),
                } );
            }
            else {
                return Err(NodePosition {
                    preds: <[Shared<Node<K, V>>; HEIGHT]>::init_with(|| {
                        let k = i;
                        i = i + 1;
                        preds[k].unwrap()
                    } ),
                    currs_or_nexts: <[Shared<Node<K, V>>; HEIGHT]>::init_with(|| {
                        let k = j;
                        j = j + 1;
                        currs[k].unwrap()
                    } ),
                } );
            }
        }
    }

    // TODO can we make the list eager with epoch? Read H&S to find out why it's lazy in the first place.
    pub fn cleanup(&self) {

    }

    pub fn insert(&self, key: K, val: V) -> bool {
        use init_with::InitWith;

        let guard = pin();
        let top_level = self.random_level();
        let mut new = Owned::new(Node::Data {
            hash: self.hash(&key),
            key: key,
            val: val,
            nexts: <[MarkableAtomic<Node<K, V>>; HEIGHT]>::init_with(|| {
                MarkableAtomic::null(false)
            } ),
            top_level: top_level,
        } );
        loop {
            match self.find_pairs(new.key(), &guard) {
                Ok(_) => {
                    return false;
                },
                Err(acc) => {
                    let mut acc = acc;

                    for lvl in (0...new.top_level()).rev() {
                        new.nexts()[lvl].store_shared(Some(acc.currs_or_nexts[lvl]), false, Ordering::SeqCst);
                    }

                    let new_ref = match acc.preds[0].nexts()[0].cas_and_ref(
                        Some(acc.currs_or_nexts[0]), false,
                        new, false,
                        Ordering::SeqCst, &guard)
                    {
                        Ok(new) => {
                            new
                        },
                        Err(new_back) => {
                            new = new_back;
                            continue;
                        },
                    };

                    for lvl in 1...top_level {
                        loop {
                            let pred = acc.preds[lvl];
                            let mut next = acc.currs_or_nexts[lvl];

                            // Update the forward pointer if it is stale
                            let new_next = new_ref.nexts()[lvl].load(Ordering::SeqCst, &guard).0.unwrap();
                            if (new_next.as_raw() != next.as_raw())
                            && !new_ref.nexts()[lvl].cas_shared(Some(new_next), false, Some(next), false, Ordering::SeqCst) {
                                break; // Give up if pointer is marked
                            }

                            if next.is_data() && next.key() == new_ref.key() {
                                next = next.nexts()[lvl].load(Ordering::SeqCst, &guard).0.unwrap();
                            }

                            if acc.preds[lvl].nexts()[lvl].cas_shared(Some(next), false, Some(new_ref), false, Ordering::SeqCst) {
                                break;
                            }

                            acc = match self.find_pairs(new_ref.key(), &guard) {
                                Ok(acc) => acc,
                                Err(acc) => acc,
                            };
                        }
                    }

                    return true;
                },
            }
        }
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        let guard = pin();
        // Maps Err to None, since if a node wasn't found there is nothing to remove
        // and then executes remove on Ok(acc: NodePosition).
        self.find_pairs(key, &guard).ok().and_then(|acc| {
            let node_to_remove: Shared<Node<K, V>> = acc.currs_or_nexts[0];
            // First mark all levels besides 0 top-to-bottom. Node is still logically in the map.
            for lvl in (1...node_to_remove.top_level()).rev() {
                let mut node_next = node_to_remove.nexts()[lvl].load(Ordering::SeqCst, &guard);
                while !node_next.1 {
                    node_to_remove.nexts()[lvl].cas_shared(node_next.0, false, node_next.0, true, Ordering::SeqCst);
                    node_next = node_to_remove.nexts()[lvl].load(Ordering::SeqCst, &guard);
                }
            }

            let mut node_next = node_to_remove.nexts()[0].load(Ordering::SeqCst, &guard);
            let mut i_marked_it = false;
            while !node_next.1 {
                // Marking the bottom level reference logically removes the node from the map. The thread which succeeds
                // in doing so must also move the value out.
                i_marked_it = node_to_remove.nexts()[0].cas_shared(node_next.0, false, node_next.0, true, Ordering::SeqCst);
                node_next = node_to_remove.nexts()[0].load(Ordering::SeqCst, &guard);
            }

            if i_marked_it {
                // TODO optimize and try to CAS acc.preds instead.
                self.find_pairs(key, &guard).is_ok();

                // We can just move it out regardless of whether it's Copy or Clone since the memory will get freed,
                // but not dropped by the epoch GC.
                // TODO shouldn't we also move out/drop the key and maybe other parts of Node<K, V>? Otherwise, anything they own will leak.
                Some(unsafe { node_to_remove.val_cpy() } )
            }
            else {
                None
            }
        } )
    }

    pub fn contains(&self, key: &K) -> bool {
        // TODO make wait-free
        let guard = pin();
        let pr = self.find_pairs(key, &guard);
        pr.is_ok()
    }

    pub fn size(&self) -> usize {
        let guard = pin();
        let mut curr: Shared<Node<K, V>> = self.head.load(Ordering::SeqCst, &guard).unwrap().nexts()[0].load(Ordering::SeqCst, &guard).0.unwrap();
        let mut count = 0;
        while curr.is_data() {
            let pr = curr.nexts()[0].load(Ordering::SeqCst, &guard);
            if !pr.1 {
                count = count + 1;
            }
            curr = pr.0.unwrap();
        }
        count
    }

    pub fn find(&self, key: &K) -> Option<V> where V: Clone {
        // TODO make wait-free
        let guard = pin();
        let pr = self.find_pairs(key, &guard);
        match pr {
            Ok(acc) => Some(acc.currs_or_nexts[0].val().clone()),
            Err(_) => None,
        }
    }

    pub fn swap(&self, key: &K, new: V) -> Option<V> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::hash::BuildHasherDefault;
    use std::collections::hash_map::DefaultHasher;

    use super::*;

    fn with_def_hasher<K, V>() -> EpochSkiplistMap<K, V, BuildHasherDefault<DefaultHasher>> where K: Eq + Hash {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        EpochSkiplistMap::with_hash_factory(build)
    }
    #[test]
    fn st_insert_1() {
        let map: EpochSkiplistMap<u32, u32, _> = with_def_hasher();
        let inserted = map.insert(0, 1);
        assert!(inserted);
        assert!(map.contains(&0));
        assert!(map.find(&0) == Some(1));
        assert!(!map.contains(&1));
    }

    #[test]
    fn st_insert_10() {
        let map: EpochSkiplistMap<u32, u32, _> = with_def_hasher();
        for i in 0..10 {
            let inserted = map.insert(i, i+1);
            assert!(inserted);
            assert!(map.contains(&i));
            assert!(map.find(&i) == Some(i+1));
            assert!(!map.contains(&(i+1)));
        }
    }

    #[test]
    fn st_remove_1() {
        let map: EpochSkiplistMap<u32, u32, _> = with_def_hasher();
        map.insert(0, 1);
        let removed = map.remove(&0);
        assert!(removed.is_some());
        assert!(removed.unwrap() == 1);
        assert!(!map.contains(&0));
        let removed_twice = map.remove(&0);
        assert!(removed_twice.is_none());
    }

    #[test]
    fn st_remove_10() {
        let map: EpochSkiplistMap<u32, u32, _> = with_def_hasher();
        for i in 0..10 {
            map.insert(i, i+1);
        }

        for i in 0..10 {
            let removed = map.remove(&i);
            assert!(removed.is_some());
            assert!(removed.unwrap() == i+1);
            assert!(!map.contains(&i));
            let removed_twice = map.remove(&i);
            assert!(removed_twice.is_none());
        }
    }

    #[test]
    fn st_size() {
        let map: EpochSkiplistMap<u32, u32, _> = with_def_hasher();
        for i in 0..10{
            assert!(map.size() as u32 == i);
            map.insert(i, i+1);
        }
    }
}
