# pylint: disable=missing-module-docstring,missing-function-docstring

from mpi4py import MPI


# pylint: disable=unused-argument
def pytest_sessionstart(session):
    if not MPI.Is_initialized():
        MPI.Init()
