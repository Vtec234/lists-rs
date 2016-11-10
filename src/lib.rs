extern crate crossbeam;

use std::sync::Arc;
use std::hash::{Hash, Hasher, BuildHasher};
use std::ops::Deref;

use crossbeam::sync::MarkableArcCell;


/// A concurrent lock-free linked list with elements sorted by the key's hash and then by the key itself.
#[derive(Debug)]
pub struct SortedLazyList<K, V, S> {
    head: Arc<Option<Node<K, V>>>,
    hasher_factory: S,
}

/// A single node in the linked list, containg a key and a value.
#[derive(Debug)]
struct Node<K, V> {
    hash: u64,
    // TODO can we make this work for unsized types?
    key: K,
    val: V,

    // A marker value of `true` indicates that this node has been logically deleted.
    next: MarkableArcCell<Option<Node<K, V>>>,
}

/// A pair of (hopefully :]) consecutive nodes for internal access.
#[derive(Debug)]
struct NodeAccess<K, V> {
    pred: Arc<Option<Node<K, V>>>,
    curr_or_next: Arc<Option<Node<K, V>>>,
}

/// Provides access to nodes in the list.
#[derive(Debug)]
pub struct SortedLazyListAccessor<K, V> {
    // This should never be None. It's only an Option so that the type is the same as in the list.
    acc: Arc<Option<Node<K, V>>>,
}

impl<K, V> SortedLazyListAccessor<K, V> {
    /// This method only has a chance to work on a recently removed node. That is, in the case when it
    /// is returned from list.remove(). Even then, it might still be co-owned by another thread. It returns
    /// the key and value if it succeeds, the same accessor otherwise.
    pub fn try_unwrap(self) -> Result<(K, V), Self> {
        match Arc::try_unwrap(self.acc) {
            Ok(opt) => {
                let unwrapped = opt.unwrap();
                Ok((unwrapped.key, unwrapped.val))
            },
            Err(arc) => Err( SortedLazyListAccessor { acc: arc, } ),
        }
    }
}

impl<K, V> Deref for SortedLazyListAccessor<K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        &(*self.acc).as_ref().unwrap().val
    }
}

impl<K, V, S> Drop for SortedLazyList<K, V, S> {
    fn drop(&mut self) {
        use std::mem;
        use std::ptr;

        unsafe {
            // First drop the actual node after `head`.
            ptr::drop_in_place(&mut Arc::get_mut(&mut self.head).unwrap().as_mut().unwrap().next as *mut MarkableArcCell<Option<Node<K, V>>>);

            // Then drop the fake head node.
            // TODO is this enough to set it to None?
            let fake_head_opt = ptr::replace(
                Arc::get_mut(&mut self.head).unwrap() as *mut Option<Node<K, V>>,
                None
            );
            mem::forget(fake_head_opt);
        }
    }
}

impl<K, V, S> SortedLazyList<K, V, S> where K: Eq + Hash, S: BuildHasher {
    pub fn with_hash_factory(f: S) -> Self {
        use std::mem;
        use std::ptr;

        // we construct an unsafe fake head node so that the types of `pred` when performing operations
        // can be uniform. surely there exists a nicer way to do hits, but i don't know it
        let mut fake_head: Node<K, V> = unsafe { mem::zeroed() };
        unsafe {
            ptr::write_bytes(
                &mut fake_head as *mut Node<K, V>,
                // the optimizer thinks zeroed memory is the None variant, so fill with 0xAB
                0xAB,
                1
            );
            ptr::write(
                &mut fake_head.next as *mut MarkableArcCell<Option<Node<K, V>>>,
                // this sentinel marks the end of the list
                MarkableArcCell::new(Arc::new(None), false)
            );
        }

        SortedLazyList {
            head: Arc::new(Some(fake_head)),
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
    /// whose hash is larger than the given key's hash and the node before it. The first node whose
    /// hash is larger than the given key's hash might be the end sentinel, which is `Arc::new(None)`.
    /// The node before it may not have the same key as the given key but may have the same hash.
    /// This method also physically removes the nodes which were marked as logically removed from
    /// the list.
    fn find_pair(&self, key: &K) -> Result<NodeAccess<K, V>, NodeAccess<K, V>> {
        let hash = self.hash(key);

        'begin_from_head: loop {
            // predecessor. the initial value of `self.head` is an invalid node and so its hash or
            // key are never checked
            let mut pred = self.head.clone();
            // currently looked at
            let mut curr = (*pred).as_ref().unwrap().next.get_arc();

            while curr.is_some() {
                // c'mon rust, this should be (succ, curr_marked) = ...
                let pr = (*curr).as_ref().unwrap().next.get();
                // successor node
                let mut succ = pr.0;
                // marked nodes are logically deleted. this routine deletes them physically
                let mut curr_marked = pr.1;

                while curr_marked {
                    // a CAS failure indicated that another thread messed with the predecessor node
                    // in which case we restart from head
                    if !(*pred).as_ref().unwrap().next.compare_exchange(
                        curr.clone(),
                        succ.clone(),
                        false,
                        false
                    ) {
			continue 'begin_from_head;
                    }

                    if (*succ).is_none() {
                        // we hit the end 
                        return Err(NodeAccess {
                            pred: pred,
                            // next
                            curr_or_next: succ,
                        } );
                    }

                    // move over the removed node. pred stays the same, curr and succ advance
                    curr = succ;
                    let pr = (*curr).as_ref().unwrap().next.get();
                    succ = pr.0;
                    curr_marked = pr.1;
                }

                let curr_hash = (*curr).as_ref().unwrap().hash;
                if curr_hash == hash
                && &(*curr).as_ref().unwrap().key == key {
                    return Ok(NodeAccess {
                        pred: pred,
                        // curr
                        curr_or_next: curr,
                    } );
                }
                else if curr_hash > hash {
                    return Err(NodeAccess {
                        pred: pred,
                        // next
                        curr_or_next: curr,
                    } );
                }

                pred = curr;
                curr = succ;
            }

	    return Err(NodeAccess {
                pred: pred,
                // next
                curr_or_next: curr,
            } );
        } // loop
    } // fn

    /// Inserts a key-value pair into the list. Fails if a node with the given key already exists.
    /// Returns whether the insert was successful.
    /// This method is obstruction-free with respect to other `remove`s and `insert`s but may wait until
    /// all `find` and `contains` call are finished.
    pub fn insert(&self, key: K, val: V) -> bool {
        let new = Arc::new( Some( Node {
            hash: self.hash(&key),
            key: key,
            val: val,

            next: MarkableArcCell::new(Arc::new(None), false),
        }));

        loop {
            match self.find_pair(&(*new).as_ref().unwrap().key) {
                Ok(_) => {
                    // a node with this key already exists
                    return false;
                },
                Err(acc) => {
                    (*new).as_ref().unwrap().next.set(acc.curr_or_next.clone(), false);
                    if (*acc.pred).as_ref().unwrap().next.compare_exchange(acc.curr_or_next, new.clone(), false, false) {
                        return true;
                    }
                    // if CAS failed, redo search
                },
            }
        }
    }

    /// If a node with the given key is found, returns an access to it, otherwise returns None.
    /// This method is wait-free.
    pub fn find(&self, key: &K) -> Option<SortedLazyListAccessor<K, V>> {
        let hash = self.hash(key);
        let mut curr = (*self.head).as_ref().unwrap().next.get_arc();

        while curr.is_some() {
            let curr_hash = (*curr).as_ref().unwrap().hash;

            if curr_hash < hash {
                curr = (*curr).as_ref().unwrap().next.get_arc();
            }
            else if curr_hash == hash
            && &(*curr).as_ref().unwrap().key == key
            && !(*curr).as_ref().unwrap().next.is_marked() {
                return Some( SortedLazyListAccessor {
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
    /// This method is obstruction-free with respect to other `remove`s and `insert`s but may wait until
    /// all `find` and `contains` call are finished.
    pub fn remove(&self, key: &K) -> Option<SortedLazyListAccessor<K, V>> {
        loop {
            match self.find_pair(key) {
                Ok(acc) => {
                    let succ = (*acc.curr_or_next).as_ref().unwrap().next.get_arc();
                    if !(*acc.curr_or_next).as_ref().unwrap().next.compare_arc_exchange_mark(succ.clone(), true) {
                        continue;
                    }

                    // wtf why is this done tiwce, i.e. here and in find_pair?
                    // possible answer: to speed things up, we try to do remove the node, but if we fail, we leave this to find_pair
                    let t = (*acc.pred).as_ref().unwrap().next.compare_exchange(acc.curr_or_next.clone(), succ, false, false);
                    // this might fail, oh well

                    return Some( SortedLazyListAccessor { acc: acc.curr_or_next, } );
                },
                Err(_) => {
                    return None;
                },
            }
        }
    }

    /// Checks whether a node with the given key is currently present in the list.
    /// This method is wait-free.
    pub fn contains(&self, key: &K) -> bool {
        self.find(key).is_some()
    }

    /// Returns the number of elements currently in the list, as observed by the thread.
    /// This method is wait-free.
    pub fn len(&self) -> usize {
        let mut curr = (*self.head).as_ref().unwrap().next.get_arc();

        let mut count = 0;
        while curr.is_some() {
            count = count + 1;
            curr = (*curr).as_ref().unwrap().next.get_arc();
        }

        count
    }
}


#[cfg(test)]
mod tests {
    use std::hash::BuildHasherDefault;
    use std::hash::SipHasher;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Barrier;
    use std::sync::atomic::Ordering;

    use crossbeam::scope;

    use super::*;

    #[test]
    fn basic_insert() {
        let build = BuildHasherDefault::<SipHasher>::default();
        let list = SortedLazyList::<u32, u32, _>::with_hash_factory(build);
        let inserted = list.insert(0, 1);
        assert!(inserted);
        assert!(list.find(&0).is_some());
        assert!(*list.find(&0).unwrap() == 1);
        assert!(list.find(&1).is_none());
    }

    #[test]
    fn basic_remove() {
        let build = BuildHasherDefault::<SipHasher>::default();
        let list = SortedLazyList::<u32, u32, _>::with_hash_factory(build);
        list.insert(0, 1);

        let removed = list.remove(&0);
        assert!(list.find(&0).is_none());
        assert!(removed.is_some());
        let acc = removed.unwrap();
        assert!(*acc == 1);
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
        let build = BuildHasherDefault::<SipHasher>::default();
        let list = SortedLazyList::<u32, u32, _>::with_hash_factory(build);
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

    const NUM_THREADS: usize = 8;

    #[test]
    fn concurrent() {
        let build = BuildHasherDefault::<SipHasher>::default();
        let list = SortedLazyList::<u32, u32, _>::with_hash_factory(build);

        let b = Barrier::new(NUM_THREADS);
        let u = AtomicUsize::new(0);
        scope(|scope| {
            for i in 0..NUM_THREADS {
                scope.spawn(|| {
                    let k = 2*u.fetch_add(1, Ordering::Relaxed) as u32;
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

        let u = AtomicUsize::new(0);
    }
}
