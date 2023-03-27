# pylint: disable=missing-module-docstring,missing-function-docstring

import mpi4py
import numba_mpi


def pytest_sessionstart(session):
    if not numba_mpi.initialized():
        mpi4py.MPI.Init()
