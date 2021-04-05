from torch.utils import data
import sys
from tqdm import tqdm
sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.data_handle import hdf5 as h5

if __name__ == "__main__":


	# Generate dataset handler
	phac2_dataset = h5.PHAC2Dataset(
		"./data/database_original.hdf5",
		"adjectives",
		fixed_length=160,
		unique = True)

	# Initialize dataset loader
	phac2 = data.DataLoader(phac2_dataset, batch_size=None,  num_workers= 1)

	# Trivial for loop
	for (ix, X) in enumerate(tqdm(phac2)):
		
		X.visualize("./entity/dataset_plot/{}.png".format(ix))
		