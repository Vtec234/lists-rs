use sync::ArcCell;
use std::sync::{Arc, Mutex, MutexGuard};
use std::hash::{Hash, Hasher, BuildHasher};


struct Entry<K, V> {
	hash: u64,
	key: K,
	val: Mutex<V>,

	next: ArcCell<Option<Entry<K, V>>>,
}

pub struct SortedListAccessor<K, V> {
	stayin_alive: Arc<Option<Entry<K, V>>>,
}

impl<K, V> SortedListAccessor<K, V> {
	pub fn lock(&self) -> MutexGuard<V> {
		(*self.stayin_alive).as_ref().unwrap().val.lock().unwrap()
	}
}

pub struct SortedList<K, V, S> {
	head: ArcCell<Option<Entry<K, V>>>,
	hasher_factory: S,
}

impl<K, V, S> SortedList<K, V, S> where K: Hash + PartialEq, S: BuildHasher {
	pub fn with_hash_factory(f: S) -> Self {
		SortedList {
			head: ArcCell::with_val(None),
			hasher_factory: f,
		}
	}

	fn hash(&self, key: &K) -> u64 {
		let mut hasher = self.hasher_factory.build_hasher();
		key.hash(&mut hasher);
		hasher.finish()
	}

	/// Inserts a key-value pair into the list. If this key is already present, updates the value in its corresponding element.
	/// May lock if somebody is holding a lock on the value to be updated. Does not update the key.
	pub fn insert(&self, key: K, val: V) {
		let hash = self.hash(&key);

		let new = Entry {
			hash: hash,
			key: key,
			val: Mutex::new(val),
			next: ArcCell::with_val(None),
		};
		let mut new_wrap = Arc::new(Some(new));

		let mut at_owned: Arc<Option<Entry<K, V>>> = Arc::new(None);
		let mut next_owned: Arc<Option<Entry<K, V>>> = self.head.get();

		loop {
			while next_owned.is_some() {
				let next_hash = (*next_owned).as_ref().unwrap().hash;

				if next_hash < hash {
					at_owned = next_owned;
					next_owned = (*at_owned).as_ref().unwrap().next.get();
				}
				else if next_hash == hash {
					{
						let next: &Entry<K, V> = (*next_owned).as_ref().unwrap();
						if next.key.eq(&(*new_wrap).as_ref().unwrap().key) {
							// there already exists and element with this key
							// change its value to the new one and return
							let val = Arc::try_unwrap(new_wrap).ok().unwrap().unwrap().val.into_inner().unwrap();
							match next.val.lock() {
								Ok(mut grd) => { *grd = val },
								// propagate panics on mutexes for now
								// TODO maybe reset mutex to be nonpoisoned?
								Err(_) => { panic!("Tried to change value of poisoned mutex."); },
							}
							return;
						}
					}

					at_owned = next_owned;
					next_owned = (*at_owned).as_ref().unwrap().next.get();
				}
				else {
					break;
				}
			}

			Arc::get_mut(&mut new_wrap).unwrap().as_mut().unwrap().next.set(next_owned.clone());
			if at_owned.is_none() {
				// new element will be the first one in list
				if self.head.compare_exchange(next_owned.clone(), new_wrap.clone()).is_ok() {
					return;
				}
			}
			else {
				if (*at_owned).as_ref().unwrap().next.compare_exchange(next_owned.clone(), new_wrap.clone()).is_ok() {
					return;
				}
			}

			/* -----------------------------------------------------------
			 * TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
			 * does this algo actually work? no idea, check for:
			 * 1. empty list
			 * 2. list with all hashes > hash
			 * 3. list with all hashes < hash
			 * 4. list wtih all hashes == hash
			 * 5. random lists
			 * -----------------------------------------------------------
			 */

			// CAS failed -> somebody snatched next_owned from under us
			// go back to traversal and find the right place again
		}
	}

	pub fn find(&self, key: &K) -> Option<SortedListAccessor<K, V>> {
		let hash = self.hash(key);

		let mut at_owned: Arc<Option<Entry<K, V>>> = self.head.get();
		while at_owned.is_some() {
			let at_hash = (*at_owned).as_ref().unwrap().hash;

			if at_hash < hash {
				at_owned = (*at_owned).as_ref().unwrap().next.get();
			}
			else if at_hash == hash {
				if (*at_owned).as_ref().unwrap().key.eq(key) {
					return Some(SortedListAccessor { stayin_alive: at_owned });
				}

				at_owned = (*at_owned).as_ref().unwrap().next.get();
			}
			else {
				return None;
			}
		}

		None
	}

	pub fn remove(&self, key: &K) -> Option<V> {
		let hash = self.hash(key);

		let mut at_owned: Arc<Option<Entry<K, V>>> = Arc::new(None);
		let mut next_owned: Arc<Option<Entry<K, V>>> = self.head.get();

		let after_victim = None;

		loop {
			while next_owned.is_some() {
				let next_hash = (*next_owned).as_ref().unwrap().hash;

				if next_hash < hash {
					at_owned = next_owned;
					next_owned = (*at_owned).as_ref().unwrap().next.get();
				}
				else if next_hash == hash {
					if (*next_owned).as_ref().unwrap().key.eq(key) {
						if at_owned.is_none() {
							if after_victim.is_none() {
								let fake = Arc::new(Some(Entry {
									hash: hash,
									key: null,
									val: Mutex::new(null),
									next: ArcCell::new(next_owned.clone()),
								}));
								after_victim = Some((*next_owned).as_ref().unwrap().next.set( TODO not at_owned but a fake entry pointing to .. where? at_owned.clone()));
							}

							// set both self.head and fake.next to after_victim to ensure no thread gets stuck in loop between fake and next or head and next
							// TODO create a fake Arc to point to for a while

							if self.head.compare_exchange(next_owned.clone(), after_victim.unwrap()).is_ok() {
								return Some(Arc::try_unwrap(next_owned).ok().unwrap().unwrap().val.into_inner().unwrap());
							}
						}
						else {
							if after_victim.is_none() {
								after_victim = Some((*next_owned).as_ref().unwrap().next.set(at_owned.clone()));
							}

							if (*at_owned).as_ref().unwrap().next.compare_exchange(next_owned.clone(), after_victim.unwrap()).is_ok() {
								return Some(Arc::try_unwrap(next_owned).ok().unwrap().unwrap().val.into_inner().unwrap());
							}
						}
					}

					at_owned = next_owned;
					next_owned = (*at_owned).as_ref().unwrap().next.get();
				}
				else {
					return None;
				}
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::hash;

	#[test]
	fn st_empty_list() {
		let a: SortedList<u32, u32, _> = SortedList::<u32, u32, _>::with_hash_factory(hash::BuildHasherDefault::<hash::SipHasher>::default());
		a.insert(20, 30);
		let lock = a.find(&20).unwrap();
		assert!(*lock.lock() == 30);
		/*
		a.remove(20);
		assert!(a.find(&20).is_none());

		a.insert(20, 40);
		let lock = a.find(&20).unwrap();
		assert!(*lock.lock() == 40);*/
	}
}