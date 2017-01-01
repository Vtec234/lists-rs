#lists-rs
This repository contains some PoC concurrent data structures:
- `ConcurrentLazySkiplist`: a lock-free concurrent map data structure implemented as a skiplist with lazy memory reclamation. based on a potential addition to `crossbeam`, the `MarkableArcCell` (see aturon/crossbeam#80)
- `ConcurrentLazyList`: as above, implemented as a linked list. prototype for the skiplist
- `ArrayofThings`: simple concurrent array, translated into Rust from http://preshing.com/20130529/a-lock-free-linear-search/
- `SortedList`: failed experiment, superseded by `ConcurrentLazySkiplist`
