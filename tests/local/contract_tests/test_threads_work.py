import numba
import numpy as np


@numba.jit(parallel=True)
def fill_array_with_thread_id(arr):
    for i in numba.prange(len(arr)):
        arr[i] = numba.get_thread_id()


def test_threads_ok():
    print(numba.get_num_threads())
    arr = np.full(100000, -1, dtype=int)
    fill_array_with_thread_id(arr)
    assert max(arr) > 0
