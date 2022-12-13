import pytest
import numba_mpi as mpi
import numpy as np


@pytest.fixture
def mpi_tmp_path(tmp_path):
    #print("INSIDE MPI_TMP_PATH")
    path_data = np.array(str(tmp_path), "c")
    path = np.empty_like(path_data) if mpi.rank() != 0 else path_data
    #print("BEFORE: mpi_tmp_path bcast()")
    mpi.bcast(path, 0)
    #print("INSIDE MPI_TMP_PATH after bcast()")
    return path.tobytes().decode("utf-8")