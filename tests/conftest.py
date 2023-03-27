# pylint: disable=missing-module-docstring,missing-function-docstring

import mpi4py


def pytest_sessionstart(_):
    mpi4py.MPI.Init()
