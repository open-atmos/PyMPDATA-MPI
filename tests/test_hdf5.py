# pylint: disable=missing-module-docstring,missing-function-docstring

from pathlib import Path

import h5py
import numba_mpi
from mpi4py import MPI


def test_hdf5(tmp_path):
    rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)

    path = Path(tmp_path) / "parallel_test.hdf5"

    file = h5py.File(path, "w", driver="mpio", comm=MPI.COMM_WORLD)

    dset = file.create_dataset("test", (numba_mpi.size(),), dtype="i")

    dset[rank] = rank

    file.close()
