import os
import sys

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import json
from datetime import datetime

import time
import mpi4py
import numba_mpi

from PyMPDATA import Options

import scenarios
from PyMPDATA_MPI.domain_decomposition import subdomain
from PyMPDATA_MPI.hdf_storage import HDFStorage
from PyMPDATA_MPI.utils import setup_dataset_and_sync_all_workers

DATASET_PATH = "datasets/"


class SimulationModel:
    def __init__(
            self,
            parameters: Optional[Dict[str, Any]] = None,
    ):
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
            print("creating dataset: " + parameters["output_datafile"])
            HDFStorage.create_dataset(
                name=parameters["dataset_name"],
                path=Path(parameters["output_datafile"]),
                grid=parameters["grid"],
                steps=parameters["output_steps"],
            )
        numba_mpi.barrier()
        self.storage = HDFStorage.mpi_context(parameters["output_datafile"], "r+", mpi4py.MPI.COMM_WORLD)

    def __call__(self):
        steps = self.parameters["output_steps"]
        x_range = slice(
            *subdomain(self.parameters["grid"][0], numba_mpi.rank(), numba_mpi.size())
        )
        print("grid: ", self.parameters["grid"], ' x_range: ', x_range)
        # start1 = time.time()
        # dataset = self.storage
        dataset = setup_dataset_and_sync_all_workers(
            self.storage, self.parameters["dataset_name"]
        )
        start = time.time()
        exec_time = self.simulation.advance(
            dataset=dataset, output_steps=steps, x_range=x_range
        )
        print("after")
        # print("exec_time: ", exec_time)
        return exec_time / 1000000, time.time() - start


def benchmark(grids):
    """
    Returns:
        dict: Dictionary containing the results. Keys are the grid dimentions, values are longest timestep in simulation
    """
    results = {}
    for i, grid in enumerate(grids):
        for k in range(2):
            # range(2) because of first run being a warmup
            model = SimulationModel(
                parameters={
                    "scenario": "CartesianScenario",
                    "mpdata_options": {"n_iters": 1},
                    "n_threads": 1,
                    "grid": grid,
                    "courant_field": (0.5, 0.5),
                    "output_steps": (36,),
                    "output_datafile": DATASET_PATH + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_output_psi_" + str(
                        i) + '_' + str(k) + ".hdf5",
                    "dataset_name": "psi",
                }
            )
            result = model()
            walltime = np.array([result[0]], dtype=np.float64).astype(np.float64)
            times_array = np.empty((numba_mpi.size(),) if numba_mpi.rank() == 0 else (0,), dtype=np.float64).astype(
                dtype=np.float64)
            numba_mpi.gather(walltime, times_array, 1, 0)
            print("exec time ", float(result[0]), ' time.time: ', result[1], " rank: ", numba_mpi.rank())
            if k == 1 and numba_mpi.rank() == 0:
                print("maximum walltime per node: ", times_array)
                results[str(grid)] = np.average(times_array)
            numba_mpi.barrier()
    return results


def type_tuple(string):
    """Convert comma-separated string to tuple."""
    return tuple(map(int, string.split(',')))


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=type_tuple, help="input grid for simulation")
    parser.add_argument('--output', type=str, help="name of .json file for saved results")
    args = parser.parse_args()
    grid = (args.grid,)
    output = args.output

    results = benchmark(grid)
    if numba_mpi.rank() == 0:
        grid = grid[0]

        if os.path.exists(output):
            with open(output, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}
        print(data)
        data[str(numba_mpi.size())] = results[str(grid)] \
            if results[str(grid)] < data.get(str(numba_mpi.size()), sys.maxsize) else data[str(numba_mpi.size())]
        print(data)
        with open(output, 'w') as f:
            json.dump(data, f)
