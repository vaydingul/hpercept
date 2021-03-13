
from torch.utils import data
import numpy as np
import torch.nn as nn
import torch
import sys
from tqdm import tqdm

sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.data_handle import hdf5 as h5
from hpercept.nn_model import cnnau, utils



phac2_dataset = h5.PHAC2Dataset(
    "./data/database_original.hdf5",
    "adjectives",
    fixed_length=160, mode="image")

phac2 = data.DataLoader(phac2_dataset, batch_size=1, num_workers=4)



model = torch.load("cnnau_model.pth")
model.eval()

encoded_pos = np.empty((len(phac2_dataset), 3))

k = 0
for (image, _) in tqdm(phac2):

	x = model.encoder(image)
	x = model.bottleneck._modules['0'](x)
	x = model.bottleneck._modules['1'](x)
	x = model.bottleneck._modules['2'](x)

	encoded_pos[k, :] = x.detach().numpy()

import matplotlib.pyplot as plt


