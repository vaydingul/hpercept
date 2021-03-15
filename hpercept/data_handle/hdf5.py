import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from . import utils


class PHAC2Dataset(data.Dataset):

	"""

	"""


	def __init__(self, fn, adj_set, fixed_length, unique = True, mode = "plain"):

		self.fn = fn

		# Open .hdf5 file to read content
		self.fh = h5py.File(self.fn, "r")

		self.mode = mode

		self.dir_set = utils.fetch_instances(self.fh, adj_set)
		
		if unique:
			_, unique_ixs = np.unique([item.split("/")[2] for item in self.dir_set], return_index=True)
			self.dir_set = self.dir_set[unique_ixs]
		
		self.fixed_length = fixed_length

	def __getitem__(self, index):

		X = utils.open_instance(self.dir_set[index], self.fh)

		X = utils.preprocess_instance(X, fixed_length = self.fixed_length)

		if self.mode == "plain":
			return X
		elif self.mode == "image":
			return torch.from_numpy(np.reshape(X.image, (1, *X.image.shape)).astype(np.float32))
		else:
			return X

	def __len__(self,):

		return len(self.dir_set)



def phac2_collate_fn(batch):
	"""

	"""

	data = [item for item in batch]


	return data



