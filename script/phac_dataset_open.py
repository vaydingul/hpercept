import sys

sys.path.insert(1, "./")
sys.path.insert(2, "./../")


from hpercept.data_handle import hdf5_new as h5
from torch.utils import data



phac2_dataset = h5.PHAC2Dataset(
    						 "./data/database_original.hdf5",
    						 "adjectives",
    						 fixed_length=160)

phac2 = data.DataLoader(phac2_dataset, batch_size = 100)

for (ix, (X)) in enumerate(phac2):

	print(X.shape)
