# pylint: disable=invalid-name,unused-argument,c-extension-no-member,too-many-arguments,line-too-long,duplicate-code

""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba
import numba_mpi as mpi
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.enumerations import SIGN_LEFT, SIGN_RIGHT

from PyMPDATA_MPI.impl.boundary_condition_commons import (
    make_scalar_boundary_condition,
    make_vector_boundary_condition,
)


class MPIPeriodic:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, size):
        # passing size insead of using mpi.size() because lack of support for non-default
        # MPI communicators. https://github.com/numba-mpi/numba-mpi/issues/64
        self.__size = size
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    def make_scalar(self, indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_scalar(
                indexers, halo, dtype, jit_flags, dimension_index
            )
        return make_scalar_boundary_condition(
            indexers,
            jit_flags,
            dimension_index,
            dtype,
            _make_get_peers(jit_flags, self.__size),
        )

    def make_vector(self, indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_vector(
                indexers, halo, dtype, jit_flags, dimension_index
            )
        return make_vector_boundary_condition(
            indexers,
            halo,
            jit_flags,
            dimension_index,
            dtype,
            _make_get_peers(jit_flags, self.__size),
        )


@lru_cache()
def _make_get_peers(jit_flags, size):
    """returns (lru-cached) numba-compiled callable."""

    @numba.njit(**jit_flags)
    def get_peers(sign):
        rank = mpi.rank()
        left_peer = (rank - 1) % size
        right_peer = (rank + 1) % size
        peers = (-1, left_peer, right_peer)
        return peers[sign], SIGN_LEFT == sign

    return get_peers
