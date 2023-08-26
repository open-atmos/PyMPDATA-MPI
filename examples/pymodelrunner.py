""" example of integration with the py-modelrunner package """
from pathlib import Path
from typing import Any, Dict, Optional

import mpi4py
import numba_mpi
from modelrunner import ModelBase, submit_job
from PyMPDATA import Options

import scenarios
from PyMPDATA_MPI.domain_decomposition import subdomain
from PyMPDATA_MPI.hdf_storage import HDFStorage
from PyMPDATA_MPI.utils import setup_dataset_and_sync_all_workers


class SimulationModel(ModelBase):
    """Class that represents model that runs simulation"""

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        *,
        strict: bool = False,
    ):
        super().__init__(parameters, output, strict=strict)
        self.parameters = parameters
        scenario_class = getattr(scenarios, parameters["scenario"])
        self.simulation = scenario_class(
            mpdata_options=Options(**parameters["mpdata_options"]),
            n_threads=parameters["n_threads"],
            grid=parameters["grid"],
            size=numba_mpi.size(),
            courant_field_multiplier=parameters["courant_field"],
            rank=numba_mpi.rank(),
        )
        if numba_mpi.rank() == 0:
            HDFStorage.create_dataset(
                name=parameters["dataset_name"],
                path=Path(parameters["output_datafile"]),
                grid=parameters["grid"],
                steps=parameters["output_steps"],
            )
        self.storage = HDFStorage.mpi_context(
            parameters["output_datafile"], "r+", mpi4py.MPI.COMM_WORLD
        )

    def __call__(self):
        steps = self.parameters["output_steps"]
        x_range = slice(
            *subdomain(self.parameters["grid"][0], numba_mpi.rank(), numba_mpi.size())
        )
        dataset = setup_dataset_and_sync_all_workers(
            self.storage, self.parameters["dataset_name"]
        )
        exec_time = self.simulation.advance(
            dataset=dataset, output_steps=steps, x_range=x_range
        )
        return exec_time


if __name__ == "__main__":
    submit_job(
        __file__,
        parameters={
            "scenario": "CartesianScenario",
            "mpdata_options": {"n_iters": 2},
            "n_threads": 1,
            "grid": (96, 96),
            "courant_field": (0.5, 0.5),
            "output_steps": tuple(i for i in range(0, 25, 2)),
            "output_datafile": "output_psi.hdf5",
            "dataset_name": "psi",
        },
        output="output_times.json",
        method="foreground",
    )
