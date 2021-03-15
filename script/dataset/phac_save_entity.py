from torch.utils import data
import sys
import numpy as np
from tqdm import tqdm


sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.data_handle import hdf5 as h5

phac2_dataset = h5.PHAC2Dataset(
	"./data/database_original.hdf5",
	"adjectives",
	fixed_length=160)

phac2 = data.DataLoader(phac2_dataset, batch_size=1, num_workers= 4, collate_fn=h5.phac2_collate_fn)

imgs = np.empty((len(phac2_dataset), 64, 160))
adjs = np.empty((len(phac2_dataset), ), dtype = np.ndarray)
names = np.empty((len(phac2_dataset), ), dtype = object)

for (ix, X) in enumerate(tqdm(phac2)):

	imgs[ix, :, :] = X[0].image
	adjs[ix] = X[0].adjective
	names[ix] = phac2_dataset.dir_set[ix].split("/")[2]

np.savez("./entity/imgs_adjs_names.npz", imgs, adjs, names)



	
			