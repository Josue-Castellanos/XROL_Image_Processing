"""Acquire data from 4D (FizCam) H5 files."""

from pathlib import Path

import numpy as np
from h5py import File

# Data layout (RxC NumPy array):
#
# [[(0, 0), (0, 1), ..., (0, C)]
#  [(1, 0), (1, 1), ..., (1, C)]
#  [   :  ,    :  ,    ,    :  ]
#  [(R, 0), (R, 1), ..., (R, C)]]
#

# Metrology data coordinate system
#
#  ^ y (pixels)
#  |              B Edge
#  |           -----------
#  |          |           |
#  |          |           |
#  |          |           |
#  |  E Edge  |           | T Edge
#  |          |           |
#  |          |           |
#  |           -----------
#  |             S Edge
#  o-------------------------------> x (pixels)


def get_wavelength_nm(file: File) -> float:
    """Return the wavelength in nm stored in the 4D H5 file."""
    return float(file["measurement0"]["genraw"].attrs["wavelength"].decode()[:-3])


def get_data_from_h5(file: File) -> np.ndarray:
    """Return the data array from the 4D H5 file."""
    return file["measurement0/genraw/data"][()]


def handle_missing_data(data: np.ndarray) -> np.ndarray:
    """Replace bad pixels with NaN."""
    data[np.where(data > 1.0e38)] = float("NaN")
    return data


def open_h5(path: Path) -> np.ndarray:
    """Return FizCam data from an H5 file as an array."""
    h5_file = File(path, mode="r")
    wavelength = get_wavelength_nm(h5_file)
    data = get_data_from_h5(h5_file)
    data = handle_missing_data(data)
    data *= wavelength
    data = np.flipud(data)
    return data
