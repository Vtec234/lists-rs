#define OBJECT_COUNT        4
#define OBJECT_MASK         (OBJECT_COUNT-1)
#define COUNT_MASK          (~OBJECT_MASK)
#define COUNT_INC           OBJECT_COUNT
#define PERSISTENT          1
#define TEMPORAL            2

typedef struct lf_object_t {
	uintptr_t volatile      rc; // "inner" counter
	lf_object_t* volatile*  back_ptr;
	void                    (*dtor)(void* obj);
} lf_object_t;

struct lf_store_t {
	uintptr_t volatile      state; // "outer" counter + index to lf_store_t::objects
	lf_object_t* volatile   objects [OBJECT_COUNT];
	CRITICAL_SECTION        write_mtx;
};

void lf_store_create(lf_store_t* store, lf_object_t* obj, void(*dtor)(void*)) {
	store->state = 0;
	InitializeCriticalSection(&store->write_mtx);
	store->objects[0] = obj;
	for (size_t i = 1; i != OBJECT_COUNT; ++i)
		store->objects[i] = 0;
	obj->rc = PERSISTENT;
	obj->back_ptr = &store->objects[0];
	obj->dtor = dtor;
}

static void lf_store_release_object(lf_object_t* obj) {
	assert(obj->rc == 0);
	assert(obj->back_ptr[0] == obj);
	obj->back_ptr[0] = 0;
	obj->dtor(obj);
}

void lf_store_destroy(lf_store_t* store) {
	uintptr_t               idx;
	lf_object_t*            obj;
	idx = store->state & OBJECT_MASK;
	obj = store->objects[idx];
	obj->rc -= (store->state & COUNT_MASK) / OBJECT_COUNT * TEMPORAL + PERSISTENT;
	lf_store_release_object(obj);
	DeleteCriticalSection(&store->write_mtx);
}

lf_object_t* lf_store_read_acquire(lf_store_t* store) {
	uintptr_t               prev;
	uintptr_t               idx;
	// completely wait-free
	// increment outer counter and simultaneously read index of the current object
	prev = (uintptr_t)_InterlockedExchangeAdd((long volatile*)&store->state, COUNT_INC);

	idx = prev & OBJECT_MASK;
	return store->objects[idx];
}

void lf_store_read_release(lf_object_t* obj) {
	uintptr_t               prev;
	// increment inner counter
	prev = (uintptr_t)_InterlockedExchangeAdd((long volatile*)&obj->rc, TEMPORAL) + TEMPORAL;

	if (prev == 0)
		lf_store_release_object(obj);
}

lf_object_t* lf_store_write_lock(lf_store_t* store) {
	uintptr_t               idx;
	EnterCriticalSection(&store->write_mtx);
	idx = store->state & OBJECT_MASK;
	return store->objects[idx];
}

void lf_store_write_unlock(lf_store_t* store, lf_object_t* obj, void(*dtor)(void*)) {
	uintptr_t               prev;
	uintptr_t               idx;
	uintptr_t               old_cnt;
	uintptr_t               old_idx;
	uintptr_t               cnt_dif;
	uintptr_t               cnt_res;
	lf_object_t*            old_obj;
	// find free object slot
	for (;;) {
		for (idx = 0; idx != OBJECT_COUNT; idx += 1) {
			if (store->objects[idx] == 0)
				break;
		}
		if (idx != OBJECT_COUNT)
			break;
		SwitchToThread();
	}
	// prepare the object for publication
	store->objects[idx] = obj;
	obj->rc = PERSISTENT;
	obj->back_ptr = &store->objects[idx];
	obj->dtor = dtor;
	// publish the object
	// and simultaneously grab previous value of the outer counter
	prev = (uintptr_t)_InterlockedExchange((long volatile*)&store->state, idx);

	old_cnt = prev & COUNT_MASK;
	old_idx = prev & OBJECT_MASK;
	old_obj = store->objects[old_idx];
	assert(old_idx != idx);
	assert(old_obj->back_ptr == &store->objects[old_idx]);
	// transfer value of the outer counter to the inner counter
	// only now object's inner counter can drop to zero
	cnt_dif = (uintptr_t)-(intptr_t)(old_cnt / OBJECT_COUNT * TEMPORAL + PERSISTENT);
	cnt_res = (uintptr_t)_InterlockedExchangeAdd((long volatile*)&old_obj->rc, cnt_dif) + cnt_dif;
	LeaveCriticalSection(&store->write_mtx);
	if (cnt_res == 0)
		lf_store_release_object(old_obj);
}
