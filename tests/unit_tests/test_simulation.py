# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name

import pytest

from PyMPDATA_MPI.domain_decomposition import mpi_indices


@pytest.mark.parametrize(
    "grid, rank, size, expected",
    (
        ((2, 3), 0, 1, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        ((2, 3), 0, 2, [[0.0, 0.0, 0.0]]),
        ((2, 3), 1, 2, [[1.0, 1.0, 1.0]]),
        ((3, 2), 0, 1, [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        ((3, 2), 0, 2, [[0.0, 0.0], [1.0, 1.0]]),
        ((3, 2), 1, 2, [[2.0, 2.0]]),
    ),
)
def test_mpi_indices(grid, rank, size, expected):
    xi, _ = mpi_indices(grid, rank, size)
    assert (xi == expected).all()
