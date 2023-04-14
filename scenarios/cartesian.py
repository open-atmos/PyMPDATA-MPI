# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name,too-few-public-methods,too-many-locals

import numpy as np
from matplotlib import pyplot
from PyMPDATA import ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic

from PyMPDATA_MPI.domain_decomposition import mpi_indices
from PyMPDATA_MPI.mpi_periodic import MPIPeriodic
from scenarios._scenario import _Scenario


class CartesianScenario(_Scenario):
    def __init__(
        self,
        *,
        mpdata_options,
        n_threads,
        grid,
        rank,
        size,
        courant_field_multiplier,
    ):
        halo = mpdata_options.n_halo

        xi, yi = mpi_indices(grid, rank, size)
        nx, ny = xi.shape

        boundary_conditions = (MPIPeriodic(size=size), Periodic())
        advectee = ScalarField(
            data=self.initial_condition(xi, yi, grid),
            halo=halo,
            boundary_conditions=boundary_conditions,
        )

        advector = VectorField(
            data=(
                np.full((nx + 1, ny), courant_field_multiplier[0]),
                np.full((nx, ny + 1), courant_field_multiplier[1]),
            ),
            halo=halo,
            boundary_conditions=boundary_conditions,
        )
        stepper = Stepper(
            options=mpdata_options,
            n_dims=2,
            n_threads=n_threads,
            left_first=rank % 2 == 0,
            # TODO #70 (see also https://github.com/open-atmos/PyMPDATA/issues/386)
            buffer_size=((ny + 2 * halo) * halo)
            * 2  # for temporary send/recv buffer on one side
            * 2,  # for complex dtype
        )
        self.solver = Solver(stepper=stepper, advectee=advectee, advector=advector)

    @staticmethod
    def initial_condition(xi, yi, grid):
        nx, ny = grid
        x0 = nx / 2
        y0 = ny / 2

        psi = np.exp(
            -((xi + 0.5 - x0) ** 2) / (2 * (nx / 10) ** 2)
            - (yi + 0.5 - y0) ** 2 / (2 * (ny / 10) ** 2)
        )
        return psi

    @staticmethod
    def quick_look(psi, zlim=(-1, 1), norm=None):
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
        cnt = ax.contourf(
            xi + 0.5,
            yi + 0.5,
            psi,
            zdir="z",
            offset=-1,
            norm=norm,
            levels=np.linspace(*zlim, 11),
        )
        cbar = pyplot.colorbar(cnt, pad=0.1, aspect=10, fraction=0.04)
        return cbar.norm
