from numpy.core.fromnumeric import repeat
from torch.utils import data
import sys
from timeit import timeit

sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.data_handle import hdf5 as h5

# Generate dataset handler
phac2_dataset = h5.PHAC2Dataset(
	"./data/database_original.hdf5",
	"adjectives",
	fixed_length=160,
	unique = True)

# Initialize dataset loader
phac2 = data.DataLoader(phac2_dataset, batch_size=None)

# Trivial for loop
for (ix, X) in enumerate(phac2):
	print(X.name)
	# do smt!		