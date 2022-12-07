# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name,too-many-locals
# based on PyMPDATA README example

import os
from pathlib import Path

import h5py
import numba_mpi as mpi
import numpy as np
from matplotlib import pyplot
from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic


class PeriodicMPI(Periodic):
    pass


def make_plot(psi, zlim, norm=None):
    # pylint: disable=invalid-name
    xi, yi = np.indices(psi.shape)
    _, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
    pyplot.gca().plot_wireframe(xi + 0.5, yi + 0.5, psi, color="red", linewidth=0.5)
    ax.set_zlim(zlim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("black")
        axis.pane.set_alpha(1)
    ax.grid(False)
    ax.set_zticks([])
    ax.set_xlabel("x/dx")
    ax.set_ylabel("y/dy")
    ax.set_proj_type("ortho")
    cnt = ax.contourf(xi + 0.5, yi + 0.5, psi, zdir="z", offset=-1, norm=norm)
    cbar = pyplot.colorbar(cnt, pad=0.1, aspect=10, fraction=0.04)
    # pylint: enable=invalid-name
    return cbar.norm


def test_2d(tmp_path, plot=True):
    # arrange
    tmp_path = Path(tmp_path) / "ground_truth.hdf5"
    ground_truth_path = (
        Path(os.path.dirname(__file__)) / "./resources/ground_truth.hdf5"
    )
    options = Options(n_iters=1)
    # TODO: define domain decomposition
    nx, ny = 24 // mpi.size(), 24

    Cx, Cy = -0.5, -0.25
    halo = options.n_halo

    # TODO: xi, yi taking into account where in the domain we are located
    xi, yi = np.indices((nx, ny), dtype=float)

    boundary_conditions = (PeriodicMPI(), PeriodicMPI())

    advectee = ScalarField(
        data=np.exp(
            -((xi + 0.5 - nx / 2) ** 2) / (2 * (nx / 10) ** 2)
            - (yi + 0.5 - ny / 2) ** 2 / (2 * (ny / 10) ** 2)
        ),
        halo=halo,
        boundary_conditions=boundary_conditions,
    )
    advector = VectorField(
        data=(np.full((nx + 1, ny), Cx), np.full((nx, ny + 1), Cy)),
        halo=halo,
        # TODO: injection of PySuperDropletLES boundary condition classes
        boundary_conditions=boundary_conditions,
    )

    stepper = Stepper(options=options, n_dims=2)

    solver = Solver(stepper=stepper, advectee=advectee, advector=advector)

    output_steps = (0, 75)

    # act
    steps_done = 0

    with h5py.File(tmp_path, "w") as file:
        print(tmp_path)
        dataset = file.create_dataset(
            "test",
            (nx, ny, len(output_steps)),  # TODO: ensure time-slices are contiguous
            dtype="float64",
            compression="gzip",
        )

        for i, output_step in enumerate(output_steps):
            n_steps = output_step - steps_done
            solver.advance(n_steps=n_steps)
            steps_done += n_steps

            tmp = dataset[:, :, i]
            tmp[:, :] = solver.advectee.get()
            dataset[:, :, i] = tmp

    # plot
    zlim = (-1, 1)
    with h5py.File(tmp_path, "r") as file, h5py.File(
        ground_truth_path, "r"
    ) as groundTruth:
        norm = make_plot(file["test"][:, :, 0], zlim)
        if plot:
            pyplot.show()
        make_plot(file["test"][:, :, 1], zlim, norm)
        if plot:
            pyplot.show()

        assert (np.array_equal(file["test"], groundTruth["test"]))
