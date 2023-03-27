# pylint: disable=missing-module-docstring,missing-function-docstring

import mpi4py
import numba_mpi as mpi
import numpy as np
import pytest


@pytest.fixture
def mpi_tmp_path(tmp_path):
    if not mpi.initialized():
        mpi4py.MPI.Init()
    path_data = np.array(str(tmp_path), "c")
    path = np.empty_like(path_data) if mpi.rank() != 0 else path_data
    mpi.bcast(path, 0)
    return path.tobytes().decode("utf-8")
