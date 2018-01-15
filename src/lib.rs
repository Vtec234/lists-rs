#![feature(alloc, heap_api, dropck_parametricity, box_syntax, test, inclusive_range_syntax)]
extern crate alloc;
extern crate rand;
extern crate crossbeam_epoch;
extern crate crossbeam_utils;
extern crate core;
extern crate test;
extern crate init_with;

//mod things_array;
mod epoch_skiplist;

//pub use things_array::*;
pub use epoch_skiplist::*;
