# pylint: disable=invalid-name
"""
PyMPDATA + numba-mpi coupler sandbox
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass
