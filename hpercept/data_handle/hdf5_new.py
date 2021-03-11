import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from . import utils


class PHAC2Dataset(data.Dataset):

	"""

	"""


	def __init__(self, fn, adj_set, fixed_length):

		self.fn = fn
		self.dir_set = utils.fetch_instances(self.fn, adj_set)
		self.fixed_length = fixed_length

	def __getitem__(self, index):

		X = utils.open_instance(self.dir_set[index][0], self.fn)

		X, y = utils.preprocess_instance(X, self.dir_set[index][1], fixed_length = self.fixed_length)

		return (torch.tensor(X.image))


	def __len__(self,):

		return len(self.dir_set)





