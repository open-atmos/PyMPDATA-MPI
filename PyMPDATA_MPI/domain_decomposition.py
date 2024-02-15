# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name

import numpy as np
from PyMPDATA.impl.domain_decomposition import make_subdomain
from PyMPDATA.impl.enumerations import OUTER

MPI_DIM = OUTER  # TODO: parametrize!

subdomain = make_subdomain(jit_flags={})


def mpi_indices(grid, rank, size):
    start, stop = subdomain(grid[MPI_DIM], rank, size)
    indices_arg = list(grid)
    indices_arg[MPI_DIM] = stop - start
    xyi = np.indices(tuple(indices_arg), dtype=float)
    xyi[MPI_DIM] += start
    return xyi
