# pylint: disable=missing-module-docstring,missing-function-docstring, missing-class-docstring
import numba_mpi
import numpy as np
import pytest
from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic

from PySuperDropletLES.mpi_boundary_condition import MPIBoundaryCondition


class TestMPIBoundaryCondition:
    @staticmethod
    @pytest.mark.skipif("numba_mpi.rank() != 0")
    @pytest.mark.parametrize(
        "bcs",
        (
            (Periodic(),),
            (
                MPIBoundaryCondition(left_neighbour_node=0, right_neighbour_node=0),
                MPIBoundaryCondition(left_neighbour_node=0, right_neighbour_node=0),
            ),
        ),
    )
    @pytest.mark.parametrize("C", (1.0, -1.0))
    @pytest.mark.parametrize("n_steps", (1,))
    @pytest.mark.parametrize("psi", ([1.0, 1.0, 2.0, 1.0, 1.0],))
    def test_1d_periodic_single_node_hardcoded(bcs, courant, n_steps, psi):
        """checks if we can emulate `Periodic` boundary condition behaviour on one node
        in one dimension with `MPIBoundaryCondition`"""

        # arrange
        grid = (len(psi),)
        options = Options(n_iters=1)
        halo = options.n_halo
        advectee = ScalarField(np.array(psi), halo=halo, boundary_conditions=bcs)
        advector = VectorField(
            (np.array([courant] * (grid[0] + 1)),), halo=halo, boundary_conditions=bcs
        )
        stepper = Stepper(options=options, grid=grid)
        solver = Solver(stepper, advectee, advector)

        # act
        solver.advance(n_steps)

        # assert
        assert abs(courant) == 1
        np.testing.assert_array_equal(
            solver.advectee.get(), np.roll(psi, int(courant) * n_steps)
        )

    @staticmethod
    def test_1d_periodic_with_mpi():
        # arrange
        _ = numba_mpi.size()
