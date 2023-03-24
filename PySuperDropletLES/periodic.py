# pylint: disable=invalid-name,unused-argument,c-extension-no-member,too-many-arguments

""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba
import numba_mpi as mpi
import numpy as np
from mpi4py import MPI
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.enumerations import SIGN_LEFT, SIGN_RIGHT

TAG = 44
comm = MPI.COMM_WORLD


class MPIPeriodic:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, size):
        self.__size = size
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    def make_scalar(self, indexers, _, __, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_scalar(indexers, _, __, jit_flags, dimension_index)
        return _make_scalar_periodic(indexers, jit_flags, dimension_index, self.__size)

    def make_vector(self, indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_vector(indexers, halo, dtype, jit_flags, dimension_index)
        return _make_vector_periodic(indexers, jit_flags, dimension_index, self.__size)


def make_send_recv(jit_flags, fill_buf):
    @numba.njit(**jit_flags)
    def _send_recv(peers, buf, psi, i_rng, j_rng, k_rng, sign, dim):
        if SIGN_LEFT == sign:
            fill_buf(buf, psi, i_rng, j_rng, k_rng, sign, dim)
            mpi.send(buf, dest=peers[sign], tag=TAG)
            mpi.recv(buf, source=peers[sign], tag=TAG)
        elif SIGN_RIGHT == sign:
            mpi.recv(buf, source=peers[sign], tag=TAG)
            fill_buf(buf, psi, i_rng, j_rng, k_rng, sign, dim)
            mpi.send(buf, dest=peers[sign], tag=TAG)

    return _send_recv


@lru_cache()
def _make_scalar_periodic(indexers, jit_flags, dimension_index, size):
    @numba.njit(**jit_flags)
    def fill_buf(buf, psi, i_rng, j_rng, k_rng, sign, _dim):
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    focus = (i, j, k)
                    buf[i - i_rng.start, j - j_rng.start, k - k_rng.start] = indexers.ats[dimension_index](
                        focus, psi, sign
                    )

    send_recv = make_send_recv(jit_flags, fill_buf)

    @numba.njit(**jit_flags)
    def fill_halos(i_rng, j_rng, k_rng, psi, span, sign):
        j_rng = range(j_rng[0], j_rng[0] + 1)
        # addressing
        rank = mpi.rank()
        peers = (-1, (rank - 1) % size, (rank + 1) % size)  # LEFT  # RIGHT

        # allocating (TODO: should not be here!)
        buf = np.empty(
            (
                len(i_rng),
                len(j_rng),
                len(k_rng),
            )
        )

        # sending/receiving
        send_recv(peers, buf, psi, i_rng, j_rng, k_rng, sign, -1)

        # writing
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    indexers.set(
                        psi,
                        i,
                        j,
                        k,
                        buf[i - i_rng.start, j - j_rng.start, k - k_rng.start],
                    )

    return fill_halos


@lru_cache()
def _make_vector_periodic(indexers, jit_flags, dimension_index, size):
    @numba.njit(**jit_flags)
    def fill_buf(buf, components, i_rng, j_rng, k_rng, sign, dim):
        parallel = dim % len(components) == dimension_index
        assert not parallel

        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    focus = (i, j, k)
                    if sign == SIGN_LEFT:
                        if dimension_index == 0:
                            # print(components[2].shape, i+1, k)
                            value = components[2][i+1, k]
                        else:
                            # print(components[2].shape, i, k+1)
                            value = components[0][i, k+1]
                        #value = indexers.atv[dimension_index](focus, components, 0, .5)
                    else:
                        if dimension_index == 0:
                            value = components[2][i-1, k]
                        else:
                            value = components[0][i, k-1]
                        # value = indexers.atv[dimension_index](focus, components, 0, -1.5)
                    buf[i - i_rng.start, j - j_rng.start, k - k_rng.start] = value
        #fprint(f"  rank={mpi.rank()} with sign={sign} is sending buf={buf}")

    send_recv = make_send_recv(jit_flags, fill_buf)

    @numba.njit(**jit_flags)
    def fill_halos_loop_vector(i_rng, j_rng, k_rng, components, dim, span, sign):
        # in 2D, j_rng is just a dummy "-44" - TODO
        j_rng = range(j_rng[0], j_rng[0] + 1)

        buf = np.empty(
            (
                len(i_rng),
                len(j_rng),
                len(k_rng),
            )
        )
        if buf.size == 0:
            return

        # addressing
        rank = mpi.rank()
        peers = (-1, (rank - 1) % size, (rank + 1) % size)  # LEFT  # RIGHT

        # send/receive
        send_recv(peers, buf, components, i_rng, j_rng, k_rng, sign, dim)

        #print(f"rank {mpi.rank()} buf_after send/recv {buf}")
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    value = buf[i - i_rng.start, j - j_rng.start, k - k_rng.start]
                    indexers.set(
                        components[dim],
                        i,
                        j,
                        k,
                        value,
                    )
    return fill_halos_loop_vector
