# pylint: disable=missing-module-docstring,missing-function-docstring,
# pylint: disable=missing-class-docstring,invalid-name,too-many-locals,too-many-arguments,c-extension-no-member
# based on PyMPDATA README example

import os
import shutil
from pathlib import Path

import mpi4py
import numba_mpi as mpi
import numpy as np
import pytest
from matplotlib import pyplot
from PyMPDATA import Options

from PyMPDATA_MPI.domain_decomposition import subdomain
from PyMPDATA_MPI.hdf_storage import HDFStorage
from PyMPDATA_MPI.utils import barrier_enclosed, setup_dataset_and_sync_all_workers
from scenarios import CartesianScenario, SphericalScenario

OPTIONS_KWARGS = (
    {"n_iters": 1},
    {"n_iters": 2, "third_order_terms": True},
    {"n_iters": 2, "nonoscillatory": True},
    {"n_iters": 3},
)

COURANT_FIELD_MULTIPLIER = (
    (0.5, 0.25),
    (-0.5, 0.25),
    (0.5, -0.25),
    (-0.5, -0.25),
)


@pytest.mark.parametrize("scenario_class", (SphericalScenario,))
@pytest.mark.parametrize("options_kwargs", OPTIONS_KWARGS)
@pytest.mark.parametrize("n_threads", (1,))  # TODO #35 : 2+
@pytest.mark.parametrize("courant_field_multiplier", COURANT_FIELD_MULTIPLIER)
def test_single_vs_multi_node(
    scenario_class,
    mpi_tmp_path_fixed,
    options_kwargs,
    n_threads,
    courant_field_multiplier,
    output_steps=range(0, 25, 2),
    grid=(64, 32),
):  # pylint: disable=redefined-outer-name
    if scenario_class is SphericalScenario and options_kwargs["n_iters"] > 1:
        pytest.skip("TODO #56")

    # if scenario_class is SphericalScenario and mpi.size() > 1:
    #     pytest.skip("TODO #56")

    plot = True and (
        "CI_PLOTS_PATH"
        in os.environ
        # and courant_field_multiplier == COURANT_FIELD_MULTIPLIER[-1]
        # and options_kwargs == OPTIONS_KWARGS[-1]
    )

    # arrange
    options_str = (
        str(options_kwargs)
        .replace(", ", "_")
        .replace(": ", ".")
        .replace("'", "")
        .replace("{", "")
        .replace("}", "")
    )
    paths = {
        mpi_max_size: mpi_tmp_path_fixed
        / f"{options_str}_mpi_max_size_{mpi_max_size}_n_threads_{n_threads}.hdf5"
        for mpi_max_size in (range(1, mpi.size() + 1))
    }

    Storage = HDFStorage
    dataset_name = "test"

    # act
    for mpi_max_size, path in paths.items():
        truncated_size = min(mpi_max_size, mpi.size())
        rank = mpi.rank()

        courant_str = (
            str(courant_field_multiplier)
            .replace(" ", "")
            .replace(",", ".")
            .replace("(", ".")
            .replace(")", ".")
        )

        plot_path = None
        if plot:
            plot_path = Path(os.environ["CI_PLOTS_PATH"]) / Path(
                f"{options_str}_rank_{mpi.rank()}_size_{mpi.size()}_c_field_{courant_str}"
            )
            shutil.rmtree(plot_path, ignore_errors=True)
            os.mkdir(plot_path)
        print("PLOT: ", plot)
        if rank == 0:
            Storage.create_dataset(
                name=dataset_name, path=path, grid=grid, steps=output_steps
            )

        with Storage.mpi_context(
            path, "r+", mpi4py.MPI.COMM_WORLD.Split(rank < truncated_size, rank)
        ) as storage:
            dataset = setup_dataset_and_sync_all_workers(storage, dataset_name)
            if rank < truncated_size:
                simulation = scenario_class(
                    mpdata_options=Options(**options_kwargs),
                    n_threads=n_threads,
                    grid=grid,
                    rank=rank,
                    size=truncated_size,
                    courant_field_multiplier=courant_field_multiplier,
                )
                x_range = slice(*subdomain(grid[0], rank, truncated_size))

                simulation.advance(dataset, output_steps, x_range)

                # plot
                if plot:
                    tmp = np.empty_like(dataset[:, :, -1])
                    for i, _ in enumerate(output_steps):
                        tmp[:] = np.nan
                        tmp[x_range, :] = dataset[x_range, :, i]
                        simulation.quick_look(tmp)
                        filename = f"step={i:04d}.svg"
                        pyplot.savefig(plot_path / filename)
                        print("Saving figure")
                        pyplot.close()

    # assert
    with barrier_enclosed():
        if mpi.rank() != 0:
            with Storage.non_mpi_contex(
                paths[1], "r"
            ) as storage_expected, Storage.non_mpi_contex(
                paths[mpi.rank() + 1], "r"
            ) as storage_actual:
                np.testing.assert_array_equal(
                    storage_expected[dataset_name][:, :, -1],
                    storage_actual[dataset_name][:, :, -1],
                )
