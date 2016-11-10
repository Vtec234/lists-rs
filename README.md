#lists-rs
This repository contains some PoC concurrent data structures:

`SortedLazyList`: a concurrent linked list sorted by key hashes based on a potential addition to `crossbeam`, `MarkableArcCell`

`array_of_things.rs`: simple concurrent array, translated into Rust from http://preshing.com/20130529/a-lock-free-linear-search/

`eager_skiplist.rs`: failed experiment, superseded by SortedLazyList
