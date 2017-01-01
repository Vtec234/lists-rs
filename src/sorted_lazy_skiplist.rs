use std::hash::{Hash, BuildHasher};
use std::sync::Arc;
use crossbeam::sync::MarkableArcCell;
use std::ops::Deref;
use rand;


#[derive(Debug)]
pub struct ConcurrentLazySkiplist<K, V, S> {
    head: Arc<Node<K, V>>,
    hasher_factory: S,
    //rng: R,
}

// the expected number of levels is log_(1/PROB)(n) for n nodes
// 16 levels is enough for 65536 nodes with PROB = 1/2
const HEIGHT: usize = 16;
const TOP_LEVEL: usize = HEIGHT - 1;

#[derive(Debug)]
enum Node<K, V> {
    Head {
        /// All references in `Head` must be some, the type is only an `Option` for convenience,
        /// since then it's the same as in `Data`.
        nexts: [Option<MarkableArcCell<Node<K, V>>>; HEIGHT],
    },
    Data {
        hash: u64,
        key: K,
        val: V,

        /// An array of references to the next node in the skiplist for each list level.
        /// References must be `Some` up to and including `top_level` and `None` above.
        /// A marker value of `true` indicates that this node has been logically deleted
        /// on the given level.
        /// A node is logically in the skiplist iff it is present and unmarked in the bottom level,
        /// other levels are only a convenience and do not matter to the user.
        // This is a statically sized array to avoid dynamic allocation and associated conurrency
        // problems. Of course, this means each node takes up more space than is strictly necessary.
        nexts: [Option<MarkableArcCell<Node<K, V>>>; HEIGHT],
        top_level: usize,
    },
    Tail,
}

impl<K, V> Node<K, V> {
    // TODO remove pattern matching when unnecessary for release builds
    fn nexts(&self) -> &[Option<MarkableArcCell<Node<K, V>>>; HEIGHT] {
        use self::Node::*;

        match self {
            &Head { ref nexts } => nexts,
            &Data { ref nexts, .. } => nexts,
            _ => panic!("6"),
        }
    }

    fn is_data(&self) -> bool {
        use self::Node::*;

        if let &Data { .. } = self { true } else { false }
    }

    fn key(&self) -> &K {
        use self::Node::*;

        if let &Data { ref key, .. } = self { key } else { panic!("3") }
    }

    fn hash(&self) -> u64 {
        use self::Node::*;

        if let &Data { hash, .. } = self { hash } else { panic!("4") }
    }

    fn top_level(&self) -> usize {
        use self::Node::*;

        match self {
            &Data { top_level, .. } => top_level,
            &Head { .. } => TOP_LEVEL,
            &Tail => TOP_LEVEL,
        }
    }
}

#[derive(Debug)]
struct NodeAccesses<K, V> {
    preds: [Arc<Node<K, V>>; HEIGHT],
    currs_or_nexts: [Arc<Node<K, V>>; HEIGHT],
}

#[derive(Debug)]
pub struct ConcurrentLazySkiplistAccessor<K, V> {
    acc: Arc<Node<K, V>>,
}

impl<K, V> ConcurrentLazySkiplistAccessor<K, V> {
    /// Unwraps the raw key and value from the accessor. This method only has a chance to work
    /// on a recently removed node. That is, when the accessor is returned from `remove()`
    /// and the node is physically removed later (either lazily or using `cleanup()`).
    /// Even then, it might still be co-owned by another thread. It returns the values
    /// if it succeeds, the same accessor otherwise.
    pub fn try_unwrap(self) -> Result<(K, V), Self> {
        match Arc::try_unwrap(self.acc) {
            Ok(node) => {
                if let Node::Data { key, val, .. } = node {
                    Ok((key, val))
                }
                else {
                    panic!("1");
                }
            },
            Err(arc) => Err(ConcurrentLazySkiplistAccessor { acc: arc, } ),
        }
    }
}

impl<K, V> Deref for ConcurrentLazySkiplistAccessor<K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        if let Node::Data { ref val, .. } = *self.acc {
            val
        }
        else {
            panic!("2");
        }
    }
}

impl<K, V, S> Drop for ConcurrentLazySkiplist<K, V, S> {
    fn drop(&mut self) {
        // TODO mitigate stack overflow for large lists. currently only mitigated for large-ish lists
        // drop by parts - iterate to 2/3 -> drop, iterate to 1/3 -> drop, etc.
        // part count depends on stack size and list len
        let len = self.len();
        const MAX_STACK: usize = 2000;
        if len > MAX_STACK {
            let mut curr = self.head.clone();
            for _ in 0..len/2 {
                curr = curr.nexts()[0].as_ref().unwrap().get_arc();
            }
            for lvl in 0...curr.top_level() {
                curr.nexts()[lvl].as_ref().unwrap().set(Arc::new(Node::Tail), false);
            }
        }
    }
}

impl<K, V, S> ConcurrentLazySkiplist<K, V, S> where K: Eq + Hash, S: BuildHasher {
    pub fn with_hash_factory(f: S) -> Self {
        use init_with::InitWith;

        let tail = Arc::new(Node::Tail);
        ConcurrentLazySkiplist {
            head: Arc::new(
                Node::Head {
                    nexts: <[Option<MarkableArcCell<Node<K, V>>>; HEIGHT]>::init_with(|| {
                        Some(MarkableArcCell::new(tail.clone(), false))
                    } )
                }
            ),
            hasher_factory: f,
        }
    }

    fn hash(&self, key: &K) -> u64 {
        use std::hash::Hasher;

        let mut hasher = self.hasher_factory.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Returns a randomly chosen level in range [0;HEIGHT), where the probability
    /// of choosing level L is PROB^(L+1).
    // PROB is currently hardwired to 1/2
    fn random_level(&self) -> usize {
        // thanks to http://ticki.github.io/blog/skip-lists-done-right/
        // TODO think about what rng to use, thread-local or not, etc
        let r: u64 = rand::random::<u64>() & (1 << HEIGHT - 1);
        for i in 0..HEIGHT {
            if (r >> i) & 1 == 1 {
                return i;
            }
        }
        return HEIGHT - 1;
    }

    fn find_pairs(&self, key: &K) -> Result<NodeAccesses<K, V>, NodeAccesses<K, V>> {
        use init_with::InitWith;

        let hash = self.hash(key);

        'begin_from_head: loop {
            let mut pred = self.head.clone();
            // TODO this is terrible. Vec can't get array'd, so use unsafe?
            let fake = Arc::new(Node::Tail);
            let mut preds = <[Arc<Node<K, V>>; HEIGHT]>::init_with(|| {
                fake.clone()
            } );
            let mut currs = <[Arc<Node<K, V>>; HEIGHT]>::init_with(|| {
                fake.clone()
            } );

            for lvl in (0...TOP_LEVEL).rev() {
                let mut curr = pred.nexts()[lvl].as_ref().unwrap().get_arc();

                while curr.is_data() {
                    let pr = curr.nexts()[lvl].as_ref().unwrap().get();
                    let succ = pr.0;
                    let curr_marked = pr.1;

                    if curr_marked {
                        if !pred.nexts()[lvl].as_ref().unwrap().compare_exchange(
                            curr.clone(),
                            succ.clone(),
                            false,
                            false
                        ) {
                            // CAS failure indicates that another thread messed with the predecessor node
                            // and we have to restart from beginning
                            continue 'begin_from_head;
                        }

                        curr = succ;
                        continue;
                    }

                    if (curr.hash() == hash && curr.key() == key) || curr.hash() > hash {
                        break;
                    }

                    pred = curr;
                    curr = succ;
                }

                preds[lvl] = pred.clone();
                currs[lvl] = curr;
            } // for

            if currs[0].is_data() && currs[0].key() == key {
                return Ok(NodeAccesses {
                    preds: preds,
                    // currs
                    currs_or_nexts: currs,
                } );
            }
            else {
                return Err(NodeAccesses {
                    preds: preds,
                    // nexts
                    currs_or_nexts: currs,
                } );
            }
        } // loop
    } // fn

    pub fn cleanup(&self) {
        for lvl in 0...TOP_LEVEL {
            'begin_from_head: loop {
                let mut pred = self.head.clone();
                let mut curr = pred.nexts()[lvl].as_ref().unwrap().get_arc();

                while curr.is_data() {
                    let pr = curr.nexts()[lvl].as_ref().unwrap().get();
                    let succ = pr.0;
                    let curr_marked = pr.1;

                    if curr_marked {
                        if !pred.nexts()[lvl].as_ref().unwrap().compare_exchange(
                            curr.clone(),
                            succ.clone(),
                            false,
                            false
                        ) {
                            continue 'begin_from_head;
                        }

                        curr = succ;
                        continue;
                    }

                    pred = curr;
                    curr = succ;
                }

                break;
            }
        }
    }

    pub fn insert(&self, key: K, val: V) -> bool {
        use init_with::InitWith;

        let top_level = self.random_level();
        let mut new = Arc::new(Node::Data {
            hash: self.hash(&key),
            key: key,
            val: val,
            top_level: top_level,

            nexts: <[Option<MarkableArcCell<Node<K, V>>>; HEIGHT]>::init_with(|| {
                None
            } ),
        } );

        loop {
            match self.find_pairs(new.key()) {
                Ok(_) => {
                    // a node with this key already exists
                    return false;
                },
                Err(acc) => {
                    let mut acc = acc;
                    {
                        let mut node = Arc::try_unwrap(new).ok().unwrap();

                        // TODO this loop goes downward unlike in H&S so that last element updated is at bottom lvl
                        // and is more likely to still be correct on CAS. is this correct?
                        for lvl in (0...top_level).rev() {
                            if let Node::Data { ref mut nexts, .. } = node {
                                nexts[lvl] = Some(MarkableArcCell::new(acc.currs_or_nexts[lvl].clone(), false));
                            }
                            else {
                                panic!("5");
                            }
                        }

                        new = Arc::new(node);
                    }

                    // a successful CAS at bottom level logically adds the node to the list. all higher levels are only for convenience
                    if !acc.preds[0].nexts()[0].as_ref().unwrap().compare_exchange(acc.currs_or_nexts[0].clone(), new.clone(), false, false) {
                        // if CAS failed, redo search
                        continue;
                    }

                    for lvl in 1...top_level {
                        loop {
                            if acc.preds[lvl].nexts()[lvl].as_ref().unwrap().compare_exchange(acc.currs_or_nexts[lvl].clone(), new.clone(), false, false) {
                                break;
                            }

                            // if inserting at given level failed we have to redo search
                            acc = match self.find_pairs(new.key()) {
                                Ok(acc) => acc,
                                // TODO can we abandon inserting as an optimization if find returns Err here?
                                Err(acc) => acc,
                            };
                        }
                    }

                    return true;
                },
            }
        }
    }

    pub fn find(&self, key: &K) -> Option<ConcurrentLazySkiplistAccessor<K, V>> {
        let hash = self.hash(key);
        let mut pred = self.head.clone();
        let mut curr = pred.nexts()[TOP_LEVEL].as_ref().unwrap().get_arc();
        for lvl in (0...TOP_LEVEL).rev() {
            curr = pred.nexts()[lvl].as_ref().unwrap().get_arc();

            while curr.is_data() {
                let pr = curr.nexts()[lvl].as_ref().unwrap().get();
                let succ = pr.0;
                let curr_marked = pr.1;

                if curr_marked {
                    curr = succ;
                    continue;
                }

                if (curr.hash() == hash && curr.key() == key) || curr.hash() > hash {
                    break;
                }

                pred = curr;
                curr = succ;
            } // while
        } // for

        if curr.is_data() && curr.key() == key {
            return Some(ConcurrentLazySkiplistAccessor {
                acc: curr,
            } );
        }
        else {
            return None;
        }
    } // fn

    pub fn remove(&self, key: &K) -> Option<ConcurrentLazySkiplistAccessor<K, V>> {
        self.find_pairs(key).ok().and_then(|acc| {
            let node_to_remove = acc.currs_or_nexts[0].clone();
            for lvl in (1...node_to_remove.top_level()).rev() {
                while !node_to_remove.nexts()[lvl].as_ref().unwrap().is_marked() {
                    node_to_remove.nexts()[lvl].as_ref().unwrap().compare_arc_exchange_mark(
                        node_to_remove.nexts()[lvl].as_ref().unwrap().get_arc(),
                        true
                    );
                }
            } // for

            let mut pr = node_to_remove.nexts()[0].as_ref().unwrap().get();
            let mut i_marked_it = false;
            while !pr.1 {
                i_marked_it = node_to_remove.nexts()[0].as_ref().unwrap().compare_exchange(
                    pr.0.clone(),
                    pr.0.clone(),
                    false,
                    true
                );
                pr = node_to_remove.nexts()[0].as_ref().unwrap().get();
            }

            if i_marked_it {
                // as an optimization, immediately try to remove physically
                for lvl in (0...node_to_remove.top_level()).rev() {
                    acc.preds[lvl].nexts()[lvl].as_ref().unwrap().compare_exchange(
                        node_to_remove.clone(),
                        node_to_remove.nexts()[lvl].as_ref().unwrap().get_arc(),
                        false,
                        false
                    );
                }

                Some(ConcurrentLazySkiplistAccessor {
                    acc: node_to_remove,
                } )
            }
            else {
                return None;
            }
        } )
    } // fn

    pub fn contains(&self, key: &K) -> bool {
        self.find(key).is_some()
    }
}

impl<K, V, S> ConcurrentLazySkiplist<K, V, S> {
    pub fn len(&self) -> usize {
        let mut curr = self.head.nexts()[0].as_ref().unwrap().get_arc();

        let mut count = 0;
        while curr.is_data() {
            if !curr.nexts()[0].as_ref().unwrap().is_marked() {
                count = count + 1;
            }
            curr = curr.nexts()[0].as_ref().unwrap().get_arc();
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
        let list = ConcurrentLazySkiplist::<u32, u32, _>::with_hash_factory(build);
        let inserted = list.insert(0, 1);
        assert!(inserted);
        assert!(list.find(&0).is_some());
        assert!(*list.find(&0).unwrap() == 1);
        assert!(list.find(&1).is_none());
    }

    #[test]
    fn basic_remove() {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        let list = ConcurrentLazySkiplist::<u32, u32, _>::with_hash_factory(build);
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
        let list = ConcurrentLazySkiplist::<u32, u32, _>::with_hash_factory(build);
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

    struct U32(u32);
    #[test]
    fn concurrent() {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        let list = ConcurrentLazySkiplist::<u32, u32, _>::with_hash_factory(build);

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
        let list = ConcurrentLazySkiplist::<u32, u32, _>::with_hash_factory(build);

        b.iter(move || {
            for i in 0..BENCH_ITERS {
                list.insert(i as u32, 1337);
            }
        } );
    }

    #[bench]
    fn bench_insert_mt(b: &mut Bencher) {
        let build = BuildHasherDefault::<DefaultHasher>::default();
        let list = ConcurrentLazySkiplist::<u32, u32, _>::with_hash_factory(build);
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
