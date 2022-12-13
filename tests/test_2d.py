# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name,too-many-locals
# based on PyMPDATA README example

from pathlib import Path
import numba_mpi as mpi
import pytest
import h5py
import numpy as np
import mpi4py
from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.impl.domain_decomposition import make_subdomain
from PyMPDATA.boundary_conditions import Periodic

from .fixtures import mpi_tmp_path as mpi_tmp_path
assert hasattr(mpi_tmp_path, "_pytestfixturefunction")

Cx, Cy = -0.5, -0.25

subdomain = make_subdomain(jit_flags={})


def mpi_indices(grid, rank, size):
    start, stop = subdomain(grid[0], rank, size)
    xi, yi = np.indices((stop - start, grid[1]), dtype=float)
    xi += start
    return xi, yi


@pytest.mark.parametrize(
    "grid, rank, size, expected", (
        ((2, 3), 0, 1, [[0., 0., 0.], [1., 1., 1.]]),

        ((2, 3), 0, 2, [[0., 0., 0.]]),
        ((2, 3), 1, 2, [[1., 1., 1.]]),

        ((3, 2), 0, 1, [[0., 0.], [1., 1.], [2., 2.]]),
        ((3, 2), 0, 2, [[0., 0.], [1., 1.]]),
        ((3, 2), 1, 2, [[2., 2.]]),
    )
)
def test_mpi_indices(grid, rank, size, expected):
    xi, _ = mpi_indices(grid, rank, size)
    assert (xi == expected).all()


def initial_condition(xi, yi, grid):
    nx, ny = grid
    return np.exp(
        -((xi + 0.5 - nx / 2) ** 2) / (2 * (nx / 10) ** 2)
        - (yi + 0.5 - ny / 2) ** 2 / (2 * (ny / 10) ** 2)
    )


@pytest.mark.parametrize("n_iters", (1, 2))
@pytest.mark.parametrize("n_threads", (1,))
def test_2d(mpi_tmp_path, n_iters, n_threads, grid=(24, 24)):  # pylint: disable=redefined-outer-name
    paths = {
        mpi_max_size: Path(mpi_tmp_path) / f"n_iters={n_iters}_mpi_max_size_{mpi_max_size}_n_threads_{n_threads}.hdf5"
        for mpi_max_size in range(1, mpi.size()+1)
    }

    for mpi_max_size, path in paths.items():
        size = min(mpi_max_size, mpi.size())
        rank = mpi.rank()

        if rank >= size:
            pass
        else:
            options = Options(n_iters=n_iters)
            halo = options.n_halo

            xi, yi = mpi_indices(grid, rank, size)
            nx, ny = xi.shape

            boundary_conditions = (Periodic(), Periodic())
            advectee = ScalarField(
                data=initial_condition(xi, yi, grid),
                halo=halo,
                boundary_conditions=boundary_conditions,
            )

            advector = VectorField(
                data=(np.full((nx + 1, ny), Cx), np.full((nx, ny + 1), Cy)),
                halo=halo,
                boundary_conditions=boundary_conditions,
            )
            stepper = Stepper(options=options, n_dims=2, n_threads=n_threads)
            solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
            output_steps = (0,)  # TODO: 75)

        if rank == 0:
            with h5py.File(path, "w") as file:
                file.create_dataset(
                    "test",
                    (*grid, len(output_steps)),
                    dtype="float64",
                )

        with h5py.File(path, "r", driver='mpio', comm=mpi4py.MPI.COMM_WORLD) as file:

            dataset = file["test"]
            if rank < size:
                steps_done = 0
                for i, output_step in enumerate(output_steps):
                    n_steps = output_step - steps_done
                    #solver.advance(n_steps=n_steps)
                    steps_done += n_steps

                    x_range = slice(*subdomain(grid[0], rank, size))
                    #dataset[x_range, :, i] = solver.advectee.get()

    mpi.barrier()

    if mpi.rank() != 0:
        with h5py.File(paths[1], "r") as file0, h5py.File(paths[mpi.rank()+1], "r") as file1:
            np.testing.assert_array_equal(file0["test"][:, :, :], file1["test"][:, :, :])


# from matplotlib import pyplot
# _, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
# pyplot.gca().plot_wireframe(xi + 0.5, yi + 0.5, advectee.get(), color="red", linewidth=0.5)
# pyplot.savefig(f"figs/size={size}_rank={rank}_maxsize={mpi_max_size}.png")
# pyplot.close()