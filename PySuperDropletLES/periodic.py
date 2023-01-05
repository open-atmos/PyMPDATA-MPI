""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba
import numba_mpi as mpi
import numpy as np
from mpi4py import MPI

from .domain_decomposition import MPI_DIM
from PyMPDATA.impl.enumerations import SIGN_LEFT, SIGN_RIGHT

TAG = 44
comm = MPI.COMM_WORLD

class MPIPeriodic:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self):
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    @staticmethod
    def make_scalar(ats, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar_periodic(ats, jit_flags)

    @staticmethod
    def make_vector(ats, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_periodic(ats, jit_flags)


@lru_cache()
def _make_scalar_periodic(ats, jit_flags):

    @numba.njit(**jit_flags)
    def fill_halos(psi, span, sign):
        rank = mpi.rank()
        size = mpi.size()

        peers = (
            -1,
            (rank - 1) % size,  # LEFT
            (rank + 1) % size  # RIGHT
        )
        #print("rank: ", mpi.rank(), " sign: ", sign, " ,index: ", sign*span)

        buf = np.full((1,), ats(*psi, sign * span))

        if SIGN_LEFT == sign:
            print("from rank: ", rank, " SENDING TO: ", peers[sign], " psi[0][mpi_dim]:", psi[0][MPI_DIM])
            comm.isend(buf, dest=peers[sign], tag=psi[0][MPI_DIM])
            comm.irecv(buf, source=peers[sign], tag=psi[0][MPI_DIM] + sign * span)
        elif SIGN_RIGHT == sign:
            print("rank: ", rank, " RECEIVING FROM: ", peers[sign], " psi[0][mpi_dim]+sign*span: ",
                  psi[0][MPI_DIM] + sign * span)
            comm.isend(buf, dest=peers[sign], tag=psi[0][MPI_DIM] + sign * span)
            comm.irecv(buf, source=peers[sign], tag=psi[0][MPI_DIM])

        print("before return, rank: ", rank, ", *psi: ", psi[0][MPI_DIM])
        return buf[0]

    return fill_halos


@lru_cache()
def _make_vector_periodic(ats, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, span, sign):
        return ats(*psi, sign * span)

    return fill_halos