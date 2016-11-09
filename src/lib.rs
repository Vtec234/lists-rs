extern crate crossbeam;

use crossbeam::sync::MarkableArcCell;
use std::sync::Arc;
use std::hash::{Hash, Hasher, BuildHasher};
use std::ops::Deref;


struct Node<K, V> {
    hash: u64,
    key: K,
    val: V,

    next: MarkableArcCell<Option<Node<K, V>>>,
}

pub struct SortedLazyList<K, V, S> {
    head: Arc<Option<Node<K, V>>>,
    hasher_factory: S,
}

struct NodeAccess<K, V> {
    pred: Arc<Option<Node<K, V>>>,
    curr: Arc<Option<Node<K, V>>>,
}

pub struct SortedLazyListAccessor<K, V> {
    acc: Arc<Option<Node<K, V>>>,
}

impl<K, V> SortedLazyListAccessor<K, V> {
    /// This method only has a chance to work on a recently removed node, so when it is returned
    /// from list.remove and not from list.find.
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
            ptr::drop_in_place(&mut Arc::get_mut(&mut self.head).unwrap().as_mut().unwrap().next as *mut MarkableArcCell<Option<Node<K, V>>>);
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

        // an unsafe head Node with invalid key, val is necessary not to require an Option<(K, V)>
        // in the Node
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

        loop {
            // a goto would be much nicer :<
            let mut breakout = false;

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
                        breakout = true;
                        break;
                    }

                    // move over the removed node. pred stays the same, curr and succ advance
                    curr = succ;
                    let pr = (*curr).as_ref().unwrap().next.get();
                    succ = pr.0;
                    curr_marked = pr.1;
                }

                if breakout {
                    break;
                }

                let curr_hash = (*curr).as_ref().unwrap().hash;
                if curr_hash == hash
                && &(*curr).as_ref().unwrap().key == key {
                    return Ok(NodeAccess {
                        pred: pred,
                        curr: curr,
                    } );
                }
                else if curr_hash > hash {
                    return Err(NodeAccess {
                        pred: pred,
                        curr: curr,
                    } );
                }

                pred = curr;
                curr = succ;
            }

            if !breakout {
                return Err(NodeAccess {
                    pred: pred,
                    curr: curr,
                } );
            } // if !breakout
        } // loop
    } // fn

    /// Inserts a key-value pair into the list. Fails if a node with the given key already exists.
    /// Returns whether the insert was successful.
    /// This method is block-free with respect to other `remove`s and `insert`s but may wait until
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
                    return false;
                },
                Err(acc) => {
                    (*new).as_ref().unwrap().next.set(acc.curr.clone(), false);
                    if (*acc.pred).as_ref().unwrap().next.compare_exchange(acc.curr, new.clone(), false, false) {
                        return true;
                    }
                },
            }
        }
    }

    /// If a node with the given key is found, returns an access to it, otherwise returns None.
    /// This method is lock-free.
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
    /// This method is block-free with respect to other `remove`s and `insert`s but may wait until
    /// all `find` and `contains` call are finished.
    pub fn remove(&self, key: &K) -> Option<SortedLazyListAccessor<K, V>> {
        loop {
            match self.find_pair(key) {
                Ok(acc) => {
                    let succ = (*acc.curr).as_ref().unwrap().next.get_arc();
                    if !(*acc.curr).as_ref().unwrap().next.compare_arc_exchange_mark(succ.clone(), true) {
                        continue;
                    }

                    // TODO why cannot this fail?
                    let t = (*acc.pred).as_ref().unwrap().next.compare_exchange(acc.curr.clone(), succ, false, false);
                    debug_assert!(t);

                    return Some( SortedLazyListAccessor { acc: acc.curr, } );
                },
                Err(_) => {
                    return None;
                },
            }
        }
    }

    /// Checks whether a node with the given key is currently present in the list.
    /// This method is lock-free.
    pub fn contains(&self, key: &K) -> bool {
        self.find(key).is_some()
    }

    /// Returns the number of elements currently in the list.
    // TODO name
    pub fn size(&self) -> usize {
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

    use super::*;
    use crossbeam::scope;

    #[test]
    fn basic() {
        let build = BuildHasherDefault::<SipHasher>::default();
        let list = SortedLazyList::<u32, u32, _>::with_hash_factory(build);
        list.insert(0, 1);
        assert!(list.find(&0).is_some());
        assert!(list.find(&1).is_none());
        list.remove(&0);
        assert!(list.find(&0).is_none());
        assert!(list.find(&1).is_none());
    }

    const NUM_THREADS: usize = 8;

    #[test]
    fn concurrent() {
        let build = BuildHasherDefault::<SipHasher>::default();
        let list = SortedLazyList::<u32, u32, _>::with_hash_factory(build);

        scope(|scope| {
            for _ in 0..NUM_THREADS {
                scope.spawn(|| {



                } );
            }
        } );

    }
}
