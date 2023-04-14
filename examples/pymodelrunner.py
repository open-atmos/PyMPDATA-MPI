""" example of integration with the py-modelrunner package """
from pathlib import Path

import numba_mpi
from modelrunner import ModelBase, submit_job
from PyMPDATA import Options

import scenarios
from PyMPDATA_MPI.domain_decomposition import subdomain
from PyMPDATA_MPI.hdf_storage import HDFStorage


class SimulationModel(ModelBase):
    def __init__(self):
        super().__init__()
        scenario_class = getattr(scenarios, self.parameters["scenario"])
        self.simulation = scenario_class(
            mpdata_options=Options(**self.parameters["mpdata_options"]),
            n_threads=self.parameters["n_threads"],
            grid=self.parameters["grid"],
            size=numba_mpi.size(),
            courant_field_multiplier=self.parameters["courant_field"],
            rank=numba_mpi.rank(),
        )
        self.storage = HDFStorage.create_dataset(
            name="psi",
            path=Path(self.output),
            grid=self.parameters["grid"],
            steps=self.parameters["output_steps"],
        )

    def __call__(self):
        steps = self.parameters["output_steps"]
        x_range = slice(
            *subdomain(self.parameters["grid"][0], numba_mpi.rank(), numba_mpi.size())
        )

        self.simulation.advance(dataset=None, output_steps=steps, x_range=x_range)


if __name__ == "__main__":
    submit_job(
        __file__,
        parameters={
            "scenario": "CartesianScenario",
            "mpdata_options": {"n_iters": 2},
            "n_threads": 1,
            "grid": (24, 24),
            "courant_field": (0.5, 0.5),
            "output_steps": range(0, 25, 2),
        },
        output="data.hdf5",
        method="foreground",
    )
