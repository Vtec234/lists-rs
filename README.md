#lists-rs
This repository contains some PoC concurrent data structures:
- `EpochSkiplistMap`: an _actually_ lock-free concurrent map data structure implemented as a skiplist with epoch-based GC memory reclamation
- `LRCSkiplistMap`: a potentially (depends on underlying primitive) lock-free concurrent map data structure implemented as a skiplist with lazy memory reclamation using reference counting, based on a potential addition to `crossbeam`, the `MarkableArcCell` (see https://github.com/aturon/crossbeam/pull/80)
- `ConcurrentLazyList`: as above, implemented as a linked list. prototype for the skiplist
- `ArrayofThings`: simple concurrent array, translated into Rust from http://preshing.com/20130529/a-lock-free-linear-search/
- `SortedList`: failed experiment, superseded by `ConcurrentLazySkiplist`
