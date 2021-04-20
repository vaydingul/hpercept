from torch.utils import data
import sys
import numpy as np
from tqdm import tqdm

# Insert custom library path to Python main lib-list
sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.data_handle import hdf5 as h5


def save_entity():
	# Generate dataset handler
	phac2_dataset = h5.PHAC2Dataset(
		"./data/database_original.hdf5",
		"adjectives",
		fixed_length=160,
		unique = False)

	# Initialize dataset loader
	phac2 = data.DataLoader(phac2_dataset, batch_size = 1,  num_workers= 1, collate_fn=h5.phac2_collate_fn)

	# Initialize empty arrays that will hold the necessary data
	imgs = np.empty((len(phac2_dataset), 64, 160))
	imgs_normalized = np.empty((len(phac2_dataset), 64, 160))
	adjs = np.empty((len(phac2_dataset), ), dtype = np.ndarray)
	names = np.empty((len(phac2_dataset), ), dtype = np.object)

	for (ix, X) in enumerate(tqdm(phac2)):
		# Allocate the arrays
		img = X[0].image
		imgs[ix, :, :] = img
		imgs_normalized[ix, :, :] = (img - np.min(img)) / (np.max(img) - np.min(img))
		adjs[ix] = X[0].adjective.astype(np.str)
		names[ix] = phac2_dataset.dir_set[ix].split("/")[2]

	# Save the whole data to .npz file
	np.savez("./entity/imgs_imgs_normalized_adjs_names.npz", imgs, imgs_normalized, adjs, names)

if __name__=="__main__":

	save_entity()


	
			