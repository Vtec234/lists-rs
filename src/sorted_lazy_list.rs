use std::sync::Arc;
use std::hash::{Hash, Hasher, BuildHasher};
use std::ops::Deref;

use crossbeam::sync::MarkableArcCell;

/// This is a concurrent lock-free singly linked list. Internally, its nodes are sorted by the key's hash.
/// It provides wait-freedom guarantees on some of its methods.
/// Most operations on this structure are O(n).
///
/// The documentation on this structure might be outdated or incorrect.
// TODO update docs
#[derive(Debug)]
// TODO explicit Send and Sync. right now it's implicitly (i think) as Send and Sync as K and V
pub struct ConcurrentLazyList<K, V, S> {
    head: Arc<Node<K, V>>,
    hasher_factory: S,
}

#[derive(Debug)]
enum Node<K, V> {
    /// The head of the list is its first node. It always exists and contains no data.
    Head {
        // TODO this could be ArcCell if we add CAS to that
        next: MarkableArcCell<Node<K, V>>,
    },
    /// A single node in the linked list, containg a key and a value.
    Data {
        hash: u64,
        // TODO can we make this work for unsized types?
        key: K,
        val: V,

        // A marker value of `true` indicates that this node has been logically deleted.
        next: MarkableArcCell<Node<K, V>>,
    },
    /// The tail is the list's last node.
    Tail
}

impl<K, V> Node<K, V> {
    fn next(&self) -> &MarkableArcCell<Node<K, V>> {
        use self::Node::*;

        match *self {
            Head { ref next } | Data { ref next, .. } => next,
            _ => panic!(),
        }
    }

    fn is_data(&self) -> bool {
        use self::Node::*;

        if let Data { .. } = *self { true } else { false }
    }

    fn key(&self) -> &K {
        use self::Node::*;

        if let Data { ref key, .. } = *self { key } else { panic!() }
    }

    fn hash(&self) -> u64 {
        use self::Node::*;

        if let Data { hash, .. } = *self { hash } else { panic!() }
    }
}

/// A pair of (hopefully :]) consecutive nodes for internal access.
#[derive(Debug)]
struct NodeAccess<K, V> {
    pred: Arc<Node<K, V>>,
    curr_or_next: Arc<Node<K, V>>,
}

/// Provides access to nodes in the list.
#[derive(Debug)]
pub struct ConcurrentLazyListAccessor<K, V> {
    acc: Arc<Node<K, V>>,
}

impl<K, V> ConcurrentLazyListAccessor<K, V> {
    /// This method only has a chance to work on a recently removed node. That is, in the case when it
    /// is returned from `remove()`` and physically removed later (either lazily or using `cleanup()`).
    /// Even then, it might still be co-owned by another thread. It returns the key and value
    /// if it succeeds, the same accessor otherwise.
    pub fn try_unwrap(self) -> Result<(K, V), Self> {
        match Arc::try_unwrap(self.acc) {
            Ok(node) => {
                if let Node::Data { key, val, .. } = node {
                    Ok((key, val))
                }
                else {
                    panic!();
                }
            },
            Err(arc) => Err(ConcurrentLazyListAccessor { acc: arc, } ),
        }
    }
}

impl<K, V> Deref for ConcurrentLazyListAccessor<K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        if let Node::Data { ref val, .. } = *self.acc {
            val
        }
        else {
            panic!();
        }
    }
}

impl<K, V, S> ConcurrentLazyList<K, V, S> where K: Eq + Hash, S: BuildHasher {
    pub fn with_hash_factory(f: S) -> Self {
        ConcurrentLazyList {
            head: Arc::new(Node::Head { next: MarkableArcCell::new(Arc::new(Node::Tail), false) } ),
            hasher_factory: f,
        }
    }

    fn hash(&self, key: &K) -> u64 {
        let mut hasher = self.hasher_factory.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// If a node with the given key was found, returns `Ok` containing that node and the node
    /// before it. If a node with the given key was not found, returns `Err` with the first node
    /// whose hash is larger than the given key's hash and the node before it.
    /// The node before it may not have the same key as the given key but may have the same hash.
    /// This method also physically removes the nodes which were marked as logically removed from
    /// the list.
    fn find_pair(&self, key: &K) -> Result<NodeAccess<K, V>, NodeAccess<K, V>> {
        let hash = self.hash(key);

        'begin_from_head: loop {
            // predecessor. the inital value is Node::Head
            let mut pred = self.head.clone();
            // currently looked at
            let mut curr = pred.next().get_arc();

            while curr.is_data() {
                // c'mon rust, this should be (succ, curr_marked) = ...
                let pr = curr.next().get();
                // successor node
                let succ = pr.0;
                // marked nodes are logically deleted. this routine deletes them physically
                let curr_marked = pr.1;

                if curr_marked {
                    if !pred.next().compare_exchange(
                        curr.clone(),
                        succ.clone(),
                        false,
                        false
                    ) {
                        // a CAS failure indicates that another thread messed with the predecessor node
                        // in which case we restart from head
                        continue 'begin_from_head;
                    }

                    // move over the removed node. pred stays the same, only curr advances
                    curr = succ;
                    continue;
                }

                if curr.hash() == hash
                && curr.key() == key {
                    return Ok(NodeAccess {
                        pred: pred,
                        // curr
                        curr_or_next: curr,
                    } );
                }
                else if curr.hash() > hash {
                    return Err(NodeAccess {
                        pred: pred,
                        // next
                        curr_or_next: curr,
                    } );
                }

                pred = curr;
                curr = succ;
            } // while

            return Err(NodeAccess {
                    pred: pred,
                    // next
                    curr_or_next: curr,
            } );
        } // loop
    } // fn

    /// Tries to physically remove as many logically removed nodes as possible. Useful if the user
    /// wants to retrieve the raw key and value from a node accessor, but the node is still
    /// owned by the list.
    pub fn cleanup(&self) {
        'begin_from_head: loop {
            // predecessor. initial value is Node::Head
            let mut pred = self.head.clone();
            // currently looked at
            let mut curr = pred.next().get_arc();

            while curr.is_data() {
                // c'mon rust, this should be (succ, curr_marked) = ...
                let pr = curr.next().get();
                // successor node
                let succ = pr.0;
                // marked nodes are logically deleted. this routine deletes them physically
                let curr_marked = pr.1;

                if curr_marked {
                    if !pred.next().compare_exchange(
                        curr.clone(),
                        succ.clone(),
                        false,
                        false
                    ) {
                        // a CAS failure indicates that another thread messed with the predecessor node
                        // in which case we restart from head
                        continue 'begin_from_head;
                    }

                    // move over the removed node. pred stays the same, only curr advances
                    curr = succ;
                    continue;
                }

                pred = curr;
                curr = succ;
            } // while

            return;
        } // loop
    } // fn

    /// Inserts a key-value pair into the list. Fails if a node with the given key already exists.
    /// Returns whether the insert was successful.
    pub fn insert(&self, key: K, val: V) -> bool {
        let new = Arc::new(Node::Data {
            hash: self.hash(&key),
            key: key,
            val: val,

            next: MarkableArcCell::new(Arc::new(Node::Tail), false),
        } );

        loop {
            match self.find_pair(new.key()) {
                Ok(_) => {
                    // a node with this key already exists
                    return false;
                },
                Err(acc) => {
                    new.next().set(acc.curr_or_next.clone(), false);
                    if acc.pred.next().compare_exchange(acc.curr_or_next, new.clone(), false, false) {
                        return true;
                    }
                    // if CAS failed, redo search
                },
            }
        }
    }

    /// If a node with the given key is found, returns an access to it, otherwise returns None.
    /// This method is wait-free.
    pub fn find(&self, key: &K) -> Option<ConcurrentLazyListAccessor<K, V>> {
        let hash = self.hash(key);
        let mut curr = self.head.next().get_arc();

        while curr.is_data() {
            if curr.hash() < hash {
                curr = curr.next().get_arc();
            }
            else if curr.hash() == hash
            && curr.key() == key
            && !curr.next().is_marked() {
                return Some(ConcurrentLazyListAccessor {
                    acc: curr,
                } );
            }
            else {
                return None;
            }
        }

        None
    }

    /// If a node with the given key is found, logically removes it from the list and returns an
    /// access to that node. Otherwise returns None.
    pub fn remove(&self, key: &K) -> Option<ConcurrentLazyListAccessor<K, V>> {
        self.find_pair(key).ok().and_then(|acc| {
            let mut pr = acc.curr_or_next.next().get();
            let mut i_marked_it = false;
            while !pr.1 {
                // TODO is it necessary to check if node still points to pr.0? maybe use next.compare_exchange_mark?
                i_marked_it = acc.curr_or_next.next().compare_exchange(pr.0.clone(), pr.0.clone(), false, true);
                pr = acc.curr_or_next.next().get();
            }

            if i_marked_it {
                // we immediately try to physically remove the node. if we fail, the node will
                // have to be removed later
                acc.pred.next().compare_exchange(acc.curr_or_next.clone(), pr.0, false, false);

                Some(ConcurrentLazyListAccessor {
                    acc: acc.curr_or_next,
                } )
            }
            else {
                None
            }
        } )
    }

    /// Checks whether a node with the given key is currently present in the list.
    /// This method is wait-free.
    pub fn contains(&self, key: &K) -> bool {
        self.find(key).is_some()
    }

    /// Returns the number of elements currently in the list, as observed by the thread.
    /// This method is wait-free.
    pub fn len(&self) -> usize {
        let mut curr = self.head.next().get_arc();

        let mut count = 0;
        while curr.is_data() {
            if !curr.next().is_marked() {
                count += 1;
            }
            curr = curr.next().get_arc();
        }

        count
    }
}


#[cfg(test)]
mod tests {
    use std::hash::BuildHasherDefault;
    use std::collections::hash_map::DefaultHasher;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    use test::Bencher;

    use crossbeam::{scope, spawn_unsafe};

    use super::*;


    #[test]
    fn basic_insert() {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        let list = ConcurrentLazyList::<u32, u32, _>::with_hash_factory(build);
        let inserted = list.insert(0, 1);
        assert!(inserted);
        assert!(list.find(&0).is_some());
        assert!(*list.find(&0).unwrap() == 1);
        assert!(list.find(&1).is_none());
    }

    #[test]
    fn basic_remove() {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        let list = ConcurrentLazyList::<u32, u32, _>::with_hash_factory(build);
        list.insert(0, 1);

        let removed = list.remove(&0);
        assert!(list.find(&0).is_none());
        assert!(removed.is_some());
        let acc = removed.unwrap();
        assert!(*acc == 1);
        list.cleanup();
        let val = acc.try_unwrap();
        assert!(val.is_ok());
        let val = val.unwrap();
        assert!(val.0 == 0);
        assert!(val.1 == 1);

        let removed_twice = list.remove(&0);
        assert!(removed_twice.is_none());
    }

    #[test]
    fn basic_len() {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        let list = ConcurrentLazyList::<u32, u32, _>::with_hash_factory(build);
        assert!(list.len() == 0);
        list.insert(0, 1);
        assert!(list.len() == 1);
        list.insert(2, 3);
        assert!(list.len() == 2);
        list.remove(&2);
        assert!(list.len() == 1);
        list.remove(&0);
        assert!(list.len() == 0);
    }

    const NUM_THREADS: usize = 4;

    // newtype wrapper that defaults to move in closures
    struct U32(u32);
    #[test]
    fn concurrent() {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        let list = ConcurrentLazyList::<u32, u32, _>::with_hash_factory(build);

        // general test of concurrent operations
        let b = Barrier::new(NUM_THREADS);
        scope(|scope| {
            for i in 0..NUM_THREADS {
                let i = U32(i as u32);
                scope.spawn(|| {
                    let k = 2*{i}.0;
                    b.wait();
                    let inserted = list.insert(k, k+1);
                    assert!(inserted);
                    let found = list.find(&k);
                    assert!(found.is_some());
                    assert!(*found.unwrap() == k+1);
                    let removed = list.remove(&k);
                    assert!(removed.is_some());
                    let acc = removed.unwrap();
                    assert!(*acc == k+1);
                    let mut val = acc.try_unwrap();
                    while let Err(a) = val {
                        list.cleanup();
                        val = a.try_unwrap();
                    }
                    let val = val.unwrap();
                    assert!(val.0 == k);
                    assert!(val.1 == k+1);

                    let removed_twice = list.remove(&k);
                    assert!(removed_twice.is_none());
                } );
            }
        } );

        assert!(list.len() == 0);

        // only one thread can remove a single item
        list.insert(20, 24);
        let u = AtomicUsize::new(0);
        scope(|scope| {
            for _ in 0..NUM_THREADS {
                scope.spawn(|| {
                    b.wait();
                    let removed = list.remove(&20);
                    if removed.is_some() {
                        u.fetch_add(1, Ordering::Relaxed);
                    }
                } );
            }
        } );
        assert!(u.load(Ordering::SeqCst) == 1);

        // cleanup works for concurrent removes
        scope(|scope| {
            for i in 0..NUM_THREADS {
                let i = U32(i as u32);
                scope.spawn(|| {
                    // TODO remove  this test from first loop and split into
                    // separate test functions
                    let k = 2*{i}.0;
                    b.wait();
                    list.insert(k, k+1);
                    let removed = list.remove(&k);
                    let acc = removed.unwrap();
                    let mut val = acc.try_unwrap();
                    while let Err(a) = val {
                        // TODO why do we have to loop?
                        // in theory one evaluation should be enough
                        // a) document lesser guarantee
                        // b) make cleanup always work
                        list.cleanup();
                        val = a.try_unwrap();
                    }
                    assert!(val.is_ok());
                } );
            }
        } );
    }

    const BENCH_ITERS: usize = 4000;

    #[bench]
    fn bench_insert_st(b: &mut Bencher) {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        let list = ConcurrentLazyList::<u32, u32, _>::with_hash_factory(build);

        b.iter(move || {
            for i in 0..BENCH_ITERS {
                list.insert(i as u32, 1337);
            }
        } );
    }

    #[bench]
    fn bench_insert_mt(b: &mut Bencher) {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        let list = ConcurrentLazyList::<u32, u32, _>::with_hash_factory(build);
        let mut handles = Vec::new();

        let count = AtomicUsize::new(NUM_THREADS);
        let val = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        // each thread waits for `flip` to flip to 1, does its thing, and decrements count
        for _ in 0..NUM_THREADS {
            handles.push(unsafe { spawn_unsafe(|| {
                let mut prev_val = 0;
                while done.load(Ordering::Relaxed) == 0 {
                    if val.load(Ordering::Relaxed) != prev_val {
                        prev_val = val.load(Ordering::Relaxed);
                        for i in 0..(BENCH_ITERS/NUM_THREADS) {
                            list.insert(i as u32, 1337);
                        }
                        count.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            } ) } );
        }

        b.iter(|| {
            val.fetch_add(1, Ordering::Relaxed);
            while count.load(Ordering::Relaxed) != 0 {}
            count.store(NUM_THREADS, Ordering::SeqCst);
        } );

        done.store(1, Ordering::Relaxed);

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
