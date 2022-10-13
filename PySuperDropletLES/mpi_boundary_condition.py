from functools import lru_cache

import numba
import numba_mpi
import numpy as np


class MPIBoundaryCondition:
    def __init__(self, left_neighbour_node, right_neighbour_node):
        self.left_neighbour_node = left_neighbour_node
        self.right_neighbour_node = right_neighbour_node

    @staticmethod
    def make_scalar(ats, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar(ats, jit_flags)

    @staticmethod
    def make_vector(ats, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector(ats, jit_flags)


@lru_cache()
def _make_scalar(ats, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, span, sign):
        # TODO
        data = np.empty(1)
        data[0] = ats(*psi, sign * span)
        status = numba_mpi.send(data, 0, 0)
        print("status send: ", status)
        data[0] = np.nan
        status_r = numba_mpi.recv(data, 0, 0)
        print("status recv: ", status_r)
        return data[0]

    return fill_halos


@lru_cache()
def _make_vector(ats, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, span, sign):
        # TODO
        data = np.empty(1)
        # data[0] = ats(*psi, sign * span)
        # status = numba_mpi.send(data, 0, sign * span)
        # print("status send: ", status)
        # data[0] = np.nan
        # status_r = numba_mpi.recv(data, 0, sign * span)
        # print("status recv: ", status_r)
        return data[0]

    return fill_halos
