# pylint: disable=missing-module-docstring,missing-function-docstring,too-many-instance-attributes,too-few-public-methods
# pylint: disable=missing-class-docstring,invalid-name,too-many-locals,too-many-arguments,c-extension-no-member, no-member

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyMPDATA import ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Polar

from PyMPDATA_MPI.domain_decomposition import mpi_indices
from PyMPDATA_MPI.mpi_periodic import MPIPeriodic
from scenarios._scenario import _Scenario

# TODO: Polar bc in PyMPDATA supports only upwind so far
OPTIONS_KWARGS = ({"n_iters": 1},)


class WilliamsonAndRasch89Settings:
    def __init__(self, *, output_steps, nlon, nlat):
        nt = output_steps[-1]

        self.output_steps = output_steps
        self.nlon = nlon
        self.nlat = nlat

        self.dlmb = 2 * np.pi / nlon
        self.dphi = np.pi / nlat

        self.r = 5 / 64 * np.pi  # original: 7/64*n.pi
        self.x0 = 3 * np.pi / 2
        self.y0 = 0

        self.udt = 2 * np.pi / nt
        self.b = -np.pi / 2
        self.h0 = 0

    def pdf(self, i, j):
        tmp = 2 * (
            (
                np.cos(self.dphi * (j + 0.5) - np.pi / 2)
                * np.sin((self.dlmb * (i + 0.5) - self.x0) / 2)
            )
            ** 2
            + np.sin((self.dphi * (j + 0.5) - np.pi / 2 - self.y0) / 2) ** 2
        )
        return self.h0 + np.where(
            # if
            tmp - self.r**2 <= 0,
            # then
            1 - np.sqrt(tmp) / self.r,
            # else
            0.0,
        )

    def ad_x(self, i, j):
        return (
            self.dlmb
            * self.udt
            * (
                np.cos(self.b) * np.cos(j * self.dphi - np.pi / 2)
                + np.sin(self.b)
                * np.sin(j * self.dphi - np.pi / 2)
                * np.cos((i + 0.5) * self.dlmb)
            )
        )

    def ad_y(self, i, j):
        return (
            -self.dlmb
            * self.udt
            * np.sin(self.b)
            * np.sin(i * self.dlmb)
            * np.cos((j + 0.5) * self.dphi - np.pi / 2)
        )

    def pdf_g_factor(self, _, y):
        return self.dlmb * self.dphi * np.cos(self.dphi * (y + 0.5) - np.pi / 2)


from PyMPDATA.impl.enumerations import INNER, OUTER


class SphericalScenario(_Scenario):
    def __init__(
        self, *, mpdata_options, n_threads, grid, rank, size, courant_field_multiplier
    ):
        self.settings = WilliamsonAndRasch89Settings(
            nlon=grid[0],  # original: 128
            nlat=grid[1],  # original: 64
            output_steps=range(0, 5120 // 3, 100),  # original: 5120
        )

        # TODO
        xi, yi = mpi_indices(grid, rank, size)
        nlon, nlat = xi.shape

        assert size == 1 or nlon < self.settings.nlon
        assert nlat == self.settings.nlat
        x0 = int(xi[0, 0])
        assert x0 == xi[0, 0]

        boundary_conditions = (
            MPIPeriodic(size=size),
            Polar(
                grid=(nlon, nlat),  # TODO ?
                longitude_idx=OUTER,
                latitude_idx=INNER,
            ),
        )

        advector_x = courant_field_multiplier[0] * np.array(
            [
                [self.settings.ad_x(i, j) for j in range(self.settings.nlat)]
                for i in range(x0, x0 + nlon + 1)
            ]
        )

        advector_y = courant_field_multiplier[1] * np.array(
            [
                [self.settings.ad_y(i, j) for j in range(self.settings.nlat + 1)]
                for i in range(x0, x0 + nlon)
            ]
        )

        advector = VectorField(
            data=(advector_x, advector_y),
            halo=mpdata_options.n_halo,
            boundary_conditions=boundary_conditions,
        )

        g_factor_z = np.array(
            [
                [self.settings.pdf_g_factor(i, j) for j in range(self.settings.nlat)]
                for i in range(x0, x0 + nlon)
            ]
        )

        # TODO: <move out>
        Cx_max = np.amax(
            np.abs((advector_x[1:, :] + advector_x[:-1, :]) / 2 / g_factor_z)
        )
        print(Cx_max)
        assert Cx_max < 1

        Cy_max = np.amax(
            np.abs((advector_y[:, 1:] + advector_y[:, :-1]) / 2 / g_factor_z)
        )
        print(Cy_max)
        assert Cy_max < 1
        # TODO: </move out>

        g_factor = ScalarField(
            data=g_factor_z,
            halo=mpdata_options.n_halo,
            boundary_conditions=boundary_conditions,
        )

        z = np.array(
            [
                [self.settings.pdf(i, j) for j in range(self.settings.nlat)]
                for i in range(x0, x0 + nlon)
            ]
        )

        advectee = ScalarField(
            data=z, halo=mpdata_options.n_halo, boundary_conditions=boundary_conditions
        )

        stepper = Stepper(options=mpdata_options, n_dims=2, non_unit_g_factor=True)
        self.solver = Solver(
            stepper=stepper, advectee=advectee, advector=advector, g_factor=g_factor
        )

    def quick_look(self, state):
        self.theta = np.linspace(0, 1, self.settings.nlat + 1, endpoint=True) * np.pi
        self.phi = np.linspace(0, 1, self.settings.nlon + 1, endpoint=True) * 2 * np.pi

        XYZ = (
            np.outer(np.sin(self.theta), np.cos(self.phi)),
            np.outer(np.sin(self.theta), np.sin(self.phi)),
            np.outer(np.cos(self.theta), np.ones(self.settings.nlon + 1)),
        )
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        norm = matplotlib.colors.Normalize(
            vmin=self.settings.h0, vmax=self.settings.h0 + 0.05
        )
        ax.plot_surface(
            *XYZ,
            rstride=1,
            cstride=1,
            facecolors=matplotlib.cm.copper_r(norm(state.T)),
            alpha=0.6,
            linewidth=0.75,
        )
        m = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.copper_r, norm=norm)
        m.set_array([])
