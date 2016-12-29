use std::hash::{Hash, BuildHasher};
use std::sync::Arc;
use crossbeam::sync::MarkableArcCell;

// 2 | 2 -- - -- - -- 4 -- - -- 9
// 1 | 2 -- - -- 3 -- 4 -- 5 -- 9
// 0 | 2 -- 2 -- 3 -- 4 -- 5 -- 9
#[derive(Debug)]
pub struct ConcurrentLazySkiplist<K, V, S> {
    head: Arc<Option<Node<K, V>>>,
    hasher_factory: S,
}

const HEIGHT: usize = 16;

#[derive(Debug)]
struct NodesAccess<K, V> {
    preds: [Arc<Option<Node<K, V>>>; HEIGHT],
    currs_or_nexts: [Arc<Option<Node<K, V>>>; HEIGHT],
}

#[derive(Debug)]
struct Node<K, V> {
    hash: u64,
    key: K,
    val: V,

    // A list of pointers to the next node for each skiplist level.
    // A marker value of true indicates that this node has been logically deleted
    // on the given level.
    //next: SomeSortOfListOrArray<MarkableArcCell<Option<Node<K, V>>>>,
    // base_list: MarkableArcCell<Option<Node<K, V>>>
    // other_lists: [MarkableArcCell<Option<Node<K, V>>>; STATIC_NUM] or List<MarkableArcCell<Option<Node<K, V>>>> <- DYNAMIC_NUM ??
    // OR
    // just [.. ; STATIC_NUM + 1] vs List<..> <- DYNAMIC_NUM + 1 (including base list)
    next: [Option<MarkableArcCell<Option<Node<K, V>>>>; HEIGHT],


}

impl<K, V, S> ConcurrentLazySkiplist<K, V, S> where K: Eq + Hash, S: BuildHasher {
    pub fn with_hash_factory(f: S) -> Self {
        use std::mem;
        use std::ptr;
        use init_with::InitWith;

        let mut head_fake: Node<K, V> = unsafe { mem::zeroed() };
        unsafe {
            ptr::write_bytes(
                &mut head_fake as *mut Node<K, V>,
                // the optimizer thinks zeroed memory is the None variant, so fill with 0xAB
                0xAB,
                1
            );
            let tail = Arc::new(None);
            let arr: [Option<MarkableArcCell<Option<Node<K, V>>>>; HEIGHT] = {
                <[Option<MarkableArcCell<Option<Node<K, V>>>>; HEIGHT]>::init_with(|| {
                    // it isn't strictly necessary for tail to be a single node rather than just
                    // an array of Arc{None}, but this is probably nicer
                    Some(MarkableArcCell::new(tail.clone(), false))
                } )
            };
            ptr::write(
                &mut head_fake.next as *mut _,
                arr
            );
        }

        ConcurrentLazySkiplist {
            head: Arc::new(Some(head_fake)),
            hasher_factory: f,
        }
    }

    fn hash(&self, key: &K) -> u64 {
        use std::hash::Hasher;

        let mut hasher = self.hasher_factory.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn find_pairs(&self, key: &K) -> Result<NodesAccess<K, V>, NodesAccess<K, V>> {
        let hash = self.hash(key);

        'begin_from_head: loop {
            let mut preds: [Arc<Option<Node<K, V>>>; HEIGHT];
            let mut currs: [Arc<Option<Node<K, V>>>; HEIGHT];



        } // loop
    } // fn
}
