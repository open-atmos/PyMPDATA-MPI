# pylint: disable=line-too-long,too-many-arguments,duplicate-code

""" polar boundary condition logic """
from functools import lru_cache

import numba
import numba_mpi as mpi
from PyMPDATA.boundary_conditions import Polar
from PyMPDATA.impl.enumerations import INNER, OUTER

from PyMPDATA_MPI.domain_decomposition import MPI_DIM
from PyMPDATA_MPI.impl.boundary_condition_commons import make_scalar_boundary_condition


class MPIPolar:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, mpi_grid, grid):
        self.worker_pool_size = grid[MPI_DIM] // mpi_grid[MPI_DIM]
        self.__mpi_size_one = self.worker_pool_size == 1

        if not self.__mpi_size_one:
            only_one_peer_per_subdomain = self.worker_pool_size % 2 == 0
            assert only_one_peer_per_subdomain

        self.polar = (
            Polar(grid=grid, longitude_idx=OUTER, latitude_idx=INNER)
            if self.__mpi_size_one
            else None
        )

    def make_scalar(self, indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""

        if self.__mpi_size_one:
            return self.polar.make_scalar(
                indexers, halo, dtype, jit_flags, dimension_index
            )
        return make_scalar_boundary_condition(
            indexers,
            jit_flags,
            dimension_index,
            dtype,
            make_get_peer(jit_flags, self.worker_pool_size),
        )

    def make_vector(self, indexers, halo, dtype, jit_flags, dimension_index):
        """TODO"""
        return Polar.make_vector(indexers, halo, dtype, jit_flags, dimension_index)


@lru_cache()
def make_get_peer(jit_flags, size):
    """returns (lru-cached) numba-compiled callable."""

    @numba.njit(**jit_flags)
    def get_peer(_):
        rank = mpi.rank()
        peer = (rank + size // 2) % size
        return peer, peer < size // 2

    return get_peer
