extern crate crossbeam_epoch;
use crossbeam_epoch::Atomic;

const OBJECT_COUNT: usize = 4;
const PERSISTENT: usize = 1;
const TEMPORAL: usize = 2;

struct StoreObj<T> {
    val: T,
    rc: Atomic,
}

struct Store {


}

fn wot() {
    
}
