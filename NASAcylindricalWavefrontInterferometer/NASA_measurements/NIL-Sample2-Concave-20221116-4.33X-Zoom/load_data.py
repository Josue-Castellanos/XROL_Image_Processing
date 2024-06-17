import h5py
import matplotlib.pyplot as plt
import numpy as np


with h5py.File(r"N:\data\fizcam\20221116-HighRi-Concave-Sample2\NIL-Sample2-Concave-20221116-1007-4.33X-Zoom-Unmasked.h5", mode="r") as h5:
    data = h5["measurement0/genraw/data"][()]

data *= 658.0  # Multiply by interferometer wavelength, units: data

plt.imshow(data)
plt.show()
