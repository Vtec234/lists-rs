use alloc::heap::{ allocate, deallocate };
use std::mem::size_of;
use std::sync::atomic::{ AtomicUsize, Ordering };
use std;

struct Entry {
    key: AtomicUsize,
	value: AtomicUsize,
}

impl Entry {
	fn new() -> Entry {
		Entry {
			key: AtomicUsize::new(0),
			value: AtomicUsize::new(0),
		}
	}
}

pub struct ArrayOfThings {
	entries: *mut Entry,
}

const ARRAY_OF_THINGS_COUNT: usize = 16384;
const PAGE_SIZE: usize = 4096;

impl ArrayOfThings {
	pub fn new() -> ArrayOfThings {
		unsafe {
			let arr = ArrayOfThings {
				entries: allocate(ARRAY_OF_THINGS_COUNT * size_of::<Entry>(), PAGE_SIZE) as *mut Entry,
			};
			let ptr = arr.entries;
			for i in 0..(ARRAY_OF_THINGS_COUNT-1) {
				std::ptr::write(
					ptr.offset(i as isize),
					Entry::new(),
				);
			}

			arr
		}
	}

	pub fn set_item(&self, key: usize, value: usize) {
		let mut at: usize = 0;
		unsafe {
			let slice = std::slice::from_raw_parts_mut(self.entries, ARRAY_OF_THINGS_COUNT);
			loop {
				let probed_key = slice[at].key.load(Ordering::Relaxed);
				if probed_key != key {
					if probed_key != 0 { at += 1; continue; }

					let prev_key = slice.get_unchecked(at).key.compare_and_swap(0, key, Ordering::Relaxed);
					if (prev_key != 0) && (prev_key != key) { at += 1; continue; }
				}

				slice.get_unchecked(at).value.store(value, Ordering::Relaxed);
				return;
			}
		}
	}

	pub fn get_item(&self, key: usize) -> usize {
		let mut at: usize = 0;
		unsafe {
			let slice = std::slice::from_raw_parts(self.entries, ARRAY_OF_THINGS_COUNT);
			loop {
				let probed_key = slice[at].key.load(Ordering::Relaxed);
				if probed_key == key { return slice.get_unchecked(at).value.load(Ordering::Relaxed); }

				if probed_key == 0 { return 0; }

				at += 1;
			}
		}
	}
}

impl Drop for ArrayOfThings {
	fn drop(&mut self) {
		unsafe {
			deallocate(self.entries as *mut u8, ARRAY_OF_THINGS_COUNT * size_of::<Entry>(), PAGE_SIZE);
		}
	}
}

unsafe impl Sync for ArrayOfThings {}
unsafe impl Send for ArrayOfThings {}


#[cfg(test)]
mod tests {
	use super::*;
	use rand::{Rng, XorShiftRng};
	use std::sync::Arc;
	use std::thread::{spawn, JoinHandle};
	use std::collections::HashMap;

	#[test]
	fn loopy_loop() {
		let arr = Arc::new(ArrayOfThings::new());
		let mut threads: Vec<JoinHandle<()>> = Vec::new();

		for range in 0..8 {
			let the_arr = arr.clone();
			threads.push(
				spawn(move || {
					let mut rand = XorShiftRng::new_unseeded();
					let mut store: HashMap<usize, usize> = HashMap::new();
					for i in 1+range*2000..2001+range*2000 {
						let mut val: usize = 0;
						while val == 0 {
							val = rand.next_u64() as usize;
						}

						store.insert(i, val);
						the_arr.set_item(i, val);
					}
					for i in 1+range*2000..2001+range*2000 {
						assert_eq!(store[&i], the_arr.get_item(i));
					}
				})
			);
		}

		for t in threads {
			t.join().unwrap();
		}
	}

	#[test]
	fn single_thread_state_check() {
		// add
		// check state
		// modify
		// check state
		// remove
		// check state
	}

	#[test]
	fn two_threads_state_check() {
		// do all of single_thread_state_check concurrently
	}

	#[test]
	fn multi_thread_spin() {
		// loop and do stuff try to find failures
	}
}
