# pylint: disable=missing-module-docstring,missing-function-docstring

from pathlib import Path

import h5py
import numba_mpi as mpi
from mpi4py import MPI
import numpy as np


def test_hdf5(tmp_path):
    rank = mpi.rank()  # The process ID (integer 0-3 for 4-process run)

    tmp_path = np.array(str(tmp_path), "c")
    tmp_path2 = np.empty_like(tmp_path)
    if rank == 0:
        tmp_path2 = tmp_path

    mpi.bcast(tmp_path2, 0)
    path = Path(tmp_path2.tobytes().decode("utf-8")) / "parallel_test.hdf5"

    with h5py.File(path, "w", driver="mpio", comm=MPI.COMM_WORLD) as file:
        dset = file.create_dataset("test", (mpi.size(),), dtype="i")
        dset[rank] = rank
        mpi.barrier()
        file.close()

    with h5py.File(path, 'r') as file:
        assert list(file["test"]) == list(range(0, mpi.size()))
