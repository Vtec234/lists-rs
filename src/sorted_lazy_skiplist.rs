use std::hash::{Hash, BuildHasher};
use std::sync::Arc;
use crossbeam::sync::MarkableArcCell;

// 2 | 2 -- - -- - -- 4 -- - -- 9
// 1 | 2 -- - -- 3 -- 4 -- 5 -- 9
// 0 | 2 -- 2 -- 3 -- 4 -- 5 -- 9
#[derive(Debug)]
pub struct SortedLazySkiplist<K, V, S> {
    head: Arc<Option<Node<K, V>>>,
    hasher_factory: S,
}

#[derive(Debug)]
struct Node<K, V> {
    hash: u64,
    key: K,
    val: V,

    // A list of pointer to the next node for each skiplist level.
    // A marker value of true indicates that this node has been logically deleted
    // on the given level.
    next: SomeSortOfListOrArray<MarkableArcCell<Option<Node<K, V>>>>,
}

impl<K, V, S> SortedLazySkiplist<K, V, S> where K: Eq + Hash, S: BuildHasher {
    pub fn with_hash_factory(f: S) -> Self {
        SortedLazySkiplist {
            head: Arc::new(None),
            hasher_factory: f,
        }
    }
}
