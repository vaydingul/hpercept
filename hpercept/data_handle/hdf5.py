import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from . import utils


class PHAC2Dataset(data.Dataset):

    """

    PHAC2Dataset reader class

    fn: Filename od the dataset to be read
    adj_set: The adjective set to be selected in the dataset ("adjective" or "adjective_neg")
    fixed_length: The downsampling size of the signals
    unique: If it is True, then the only unique 600 samples will be returned, if it is not,
                    then, the whole dataset will be loaded.
    mode: If it is "plain", then the PHAC2 instance will be returned in each minibatch,
              if it is "image", then the PHAC2.image attribute will be returned in each minibatch.


    """

    def __init__(self, fn, adj_set, fixed_length, unique=True, mode="plain"):

        # Filename
        self.fn = fn

        # Open .hdf5 file to read content
        self.fh = h5py.File(self.fn, "r")
        #! RETURN TO THE OLD FORM, WHERE FILENAME IS SHARED,
        #! RATHER THAN FILE HANDLE ITSELF. IT PREVENTS MULTI-THREADING

        # Batching mode
        self.mode = mode

        # Directory set in the dataset
        self.dir_set = utils.fetch_instances(self.fh, adj_set)

        # If unique is True, then it returns the unique data points
        if unique:
            _, unique_ixs = np.unique([item.split("/")[2]
                                       for item in self.dir_set], return_index=True)
            self.dir_set = self.dir_set[unique_ixs]

        # Signal cropping length
        self.fixed_length = fixed_length

    def __getitem__(self, index):
        """

        Read the specified item and apply preprocessing on that

        """

        # Read the content of the directory
        X = utils.open_instance(self.dir_set[index], self.fh)
        # Apply preprocess operations on the datapoint
        X = utils.preprocess_instance(X, fixed_length=self.fixed_length)

        if self.mode == "plain":
            return X
        elif self.mode == "image":
            return torch.from_numpy(np.reshape(X.image, (1, *X.image.shape)).astype(np.float32))
        else:
            return X

    def __len__(self,):
        """

        The length of the dataset equals to the number of datapoints to be sampled in total

        """
        return len(self.dir_set)


def phac2_collate_fn(batch):
    """

    Trivial collate function, since it is not default in Torch.

    """

    data = [item for item in batch]

    return data
