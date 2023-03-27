# pylint: disable=missing-module-docstring,missing-function-docstring


# pylint: disable=unused-argument,unused-import,import-outside-toplevel
def pytest_sessionstart(session):
    from mpi4py import MPI
