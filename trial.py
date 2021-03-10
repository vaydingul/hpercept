import h5py as h5
import numpy as np

f = h5.File("./data/database_original.hdf5", "r")

print(np.array(f["adjectives/bumpy/bubble_wrap_214_01/accelerometer"]))



print(list(f["adjectives"].keys()))
