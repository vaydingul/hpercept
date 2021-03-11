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
		
		self.mode = mode

		self.dir_set = utils.fetch_instances(self.fn, adj_set)
		
		if unique:
			_, unique_ixs = np.unique([item[0].split("/")[2] for item in self.dir_set], return_index=True)
			self.dir_set = self.dir_set[unique_ixs]
		
		self.fixed_length = fixed_length

	def __getitem__(self, index):

		X = utils.open_instance(self.dir_set[index][0], self.fn)

		X, y = utils.preprocess_instance(X, self.dir_set[index][1], fixed_length = self.fixed_length)

		if self.mode == "plain":
			return (X, y)
		elif self.mode == "image":
			return (torch.from_numpy(np.reshape(X.image, (1, *X.image.shape)).astype(np.float32)), y)
		else:
			return (X, y)

	def __len__(self,):

		return len(self.dir_set)



def phac2_collate_fn(batch):
	"""

	"""

	data = [item[0] for item in batch]
	target = [item[1] for item in batch]

	return data, target



