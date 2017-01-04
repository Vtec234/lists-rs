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
    // PROB is currently hardwired to 1/2
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

        'begin_from_head: loop {
            // TODO this shouldn't be Option, it's annoying
            let mut preds = <[Option<Shared<Node<K, V>>>; HEIGHT]>::init_with(|| {
                None
            } );
            let mut currs = <[Option<Shared<Node<K, V>>>; HEIGHT]>::init_with(|| {
                None
            } );

            let mut pred: Shared<Node<K, V>> = self.head.load(Ordering::SeqCst, g).unwrap();
            for lvl in (0...TOP_LEVEL).rev() {
                let mut pred_next: (Option<Shared<Node<K, V>>>, bool) = pred.nexts()[lvl].load(Ordering::SeqCst, g);
                if pred_next.1 {
                    // TODO why?
                    // retry if predecessor is marked
                    continue 'begin_from_head;
                }

                /* Find unmarked node pair at this level. */
                let mut curr: Shared<Node<K, V>> = pred_next.0.unwrap();
                while curr.is_data() {
                    // Skip a sequence of marked nodes
                    let mut curr_next: (Option<Shared<Node<K, V>>>, bool) = curr.nexts()[lvl].load(Ordering::SeqCst, g);
                    while curr.is_data() {
                        curr_next = curr.nexts()[lvl].load(Ordering::SeqCst, g);
                        if !curr_next.1 {
                            // curr is unmarked
                            break;
                        }
                        // curr is marked, step over and keep pred
                        curr = curr_next.0.unwrap();
                    }

                    if (curr.hash() == hash && curr.key() == key) || curr.hash() > hash {
                        break;
                    }

                    pred = curr;
                    // wat
                    pred_next = curr_next;
                    curr = curr_next.0.unwrap();
                }

                /* Ensure left and right nodes are adjacent. */
                if (pred_next.0.unwrap().as_raw() != curr.as_raw())
                    && !pred.nexts()[lvl].cas_shared(
                        pred_next.0, pred_next.1,
                        Some(curr), pred_next.1,
                        Ordering::SeqCst)
                {
                    continue 'begin_from_head;
                }

                preds[lvl] = Some(pred);
                currs[lvl] = Some(curr);
            }

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
                        preds[k].unwrap()
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
                        preds[k].unwrap()
                    } ),
                } );
            }
        }
    }

    // TODO can we make the list eager with epoch? read H&S to find out why it's lazy in the first place
    pub fn cleanup(&self) {

    }

    pub fn insert(&self, key: K, val: V) -> bool {
        use init_with::InitWith;

        let guard = pin();
        let new = Owned::new(Node::Data {
            hash: self.hash(&key),
            key: key,
            val: val,
            nexts: <[MarkableAtomic<Node<K, V>>; HEIGHT]>::init_with(|| {
                MarkableAtomic::null(false)
            } ),
            top_level: self.random_level(),
        } );
        loop {
            match self.find_pairs(new.key(), &guard) {
                Ok(_) => {
                    return false;
                },
                Err(acc) => {
                    let node: Shared<Node<K, V>> = acc.currs_or_nexts[0];
                    for lvl in 0...node.top_level() {}

                    return false;
                },
            }
        }
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        let guard = pin();
        loop {
            match self.find_pairs(key, &guard) {
                Ok(acc) => {
                    let node_to_remove: Shared<Node<K, V>> = acc.currs_or_nexts[0];
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
                        i_marked_it = node_to_remove.nexts()[0].cas_shared(node_next.0, false, node_next.0, false, Ordering::SeqCst);
                        node_next = node_to_remove.nexts()[0].load(Ordering::SeqCst, &guard);
                    }

                    if i_marked_it {
                        // TODO optimize and try to cas acc.preds instead
                        self.find_pairs(key, &guard);

                        // we can just move it out regardless of whether it's Copy or Clone since the memory will get freed, but not dropped
                        return Some(unsafe { node_to_remove.val_cpy() } );
                    }
                    else {
                        return None;
                    }
                },
                Err(_) => {
                    return None;
                }
            }
        }
    }

    pub fn contains(&self, key: &K) -> bool {
        // TODO make wait-free
        let guard = pin();
        let pr = self.find_pairs(key, &guard);
        pr.is_ok()
    }

    pub fn size(&self) -> usize {
        0
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
