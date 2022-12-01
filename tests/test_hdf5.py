# pylint: disable=missing-module-docstring,missing-function-docstring

from pathlib import Path

import h5py
import numba
import numba_mpi as mpi
import numpy as np
from mpi4py import MPI


@numba.njit()
def step(dset):
    rank = mpi.rank()
    dset[rank] = rank
    mpi.barrier()


def test_hdf5(tmp_path):
    rank = mpi.rank()  # The process ID (integer 0-3 for 4-process run)

    path_data = np.array(str(tmp_path), "c")
    path = np.empty_like(path_data) if rank != 0 else path_data

    mpi.bcast(path, 0)

    path = Path(path.tobytes().decode("utf-8")) / "parallel_test.hdf5"

    with h5py.File(path, "w", driver="mpio", comm=MPI.COMM_WORLD) as file:
        dset = file.create_dataset("test", (mpi.size(),), dtype="i")

        tmp = dset[:]
        step(tmp)
        dset[:] = tmp

        file.close()

    with h5py.File(path, "r") as file:
        assert list(file["test"]) == list(range(0, mpi.size()))
