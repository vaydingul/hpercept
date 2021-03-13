from torch.utils import data
from sklearn.manifold import MDS
import sys
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.data_handle import hdf5 as h5

phac2_dataset = h5.PHAC2Dataset(
    "./data/database_original.hdf5",
    "adjectives",
    fixed_length=160)

phac2 = data.DataLoader(phac2_dataset, batch_size=None, num_workers=4)

imgs = [X.image for (X, _) in tqdm(phac2)]

for k in range(0,100, 10):
	plt.figure()
	plt.imshow(imgs[k], cmap='hot', interpolation='nearest')

plt.show()
