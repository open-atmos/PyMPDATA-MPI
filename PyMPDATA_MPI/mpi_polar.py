from functools import lru_cache

import numba
import numba_mpi as mpi
from PyMPDATA.boundary_conditions import Polar
from PyMPDATA.impl.enumerations import (
    INNER,
    INVALID_INDEX,
    OUTER,
    SIGN_LEFT,
    SIGN_RIGHT,
)

from PyMPDATA_MPI.domain_decomposition import MPI_DIM

IRRELEVANT = 666


class MPIPolar:
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
        return _make_scalar_periodic(
            indexers, jit_flags, dimension_index, self.worker_pool_size, dtype
        )

    def make_vector(self, indexers, halo, dtype, jit_flags, dimension_index):
        """TODO"""
        return Polar.make_vector(indexers, halo, dtype, jit_flags, dimension_index)


@lru_cache()
def _make_scalar_periodic(
    indexers, jit_flags, dimension_index, worker_pool_size, dtype
):
    @numba.njit(**jit_flags)
    def fill_buf(buf, psi, i_rng, k_rng, sign, _dim):
        for i in i_rng:
            for k in k_rng:
                buf[i - i_rng.start, k - k_rng.start] = indexers.ats[dimension_index](
                    (i, INVALID_INDEX, k), psi, sign
                )

    send_recv = _make_send_recv(
        indexers.set, jit_flags, fill_buf, worker_pool_size, dtype
    )

    @numba.njit(**jit_flags)
    def fill_halos(buffer, i_rng, j_rng, k_rng, psi, _, sign):
        send_recv(buffer, psi, i_rng, j_rng, k_rng, sign, IRRELEVANT, psi)

    return fill_halos


def _make_send_recv(set_value, jit_flags, fill_buf, size, dtype):
    @numba.njit(**jit_flags)
    def get_buffer_chunk(buffer, i_rng, k_rng, chunk_index):
        chunk_size = len(i_rng) * len(k_rng)
        return buffer.view(dtype)[
            chunk_index * chunk_size : (chunk_index + 1) * chunk_size
        ].reshape((len(i_rng), len(k_rng)))

    @numba.njit(**jit_flags)
    def get_peer():
        rank = mpi.rank()
        peer = (rank + size // 2) % size
        return peer

    @numba.njit(**jit_flags)
    def fill_output(output, buffer, i_rng, j_rng, k_rng):
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    set_value(
                        output,
                        i,
                        j,
                        k,
                        buffer[i - i_rng.start, k - k_rng.start],
                    )

    @numba.njit(**jit_flags)
    def _send(buf, peer, fill_buf_args):
        fill_buf(buf, *fill_buf_args)
        mpi.send(buf, dest=peer)

    @numba.njit(**jit_flags)
    def _recv(buf, peer):
        mpi.recv(buf, source=peer)

    @numba.njit(**jit_flags)
    def _send_recv(buffer, psi, i_rng, j_rng, k_rng, sign, dim, output):
        buf = get_buffer_chunk(buffer, i_rng, k_rng, chunk_index=0)
        peer = get_peer()
        fill_buf_args = (psi, i_rng, k_rng, sign, dim)

        if peer < size // 2:
            _send(buf=buf, peer=peer, fill_buf_args=fill_buf_args)
            _recv(buf=buf, peer=peer)
        else:
            _recv(buf=buf, peer=peer)
            tmp = get_buffer_chunk(buffer, i_rng, k_rng, chunk_index=1)
            _send(buf=tmp, peer=peer, fill_buf_args=fill_buf_args)

        fill_output(output, buf, i_rng, j_rng, k_rng)

    return _send_recv
