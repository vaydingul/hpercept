from torch.utils import data
from sklearn.manifold import MDS
import sys
from skimage.feature import hog
import numpy as np
sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.data_handle import hdf5 as h5

phac2_dataset = h5.PHAC2Dataset(
    "./data/database_original.hdf5",
    "adjectives",
    fixed_length=160)

phac2 = data.DataLoader(phac2_dataset, batch_size=None, num_workers=4)

hog_vals = [hog(X.image, orientations=5, pixels_per_cell=(32, 32), cells_per_block=(1,1)) for (X, _) in phac2]

# It should be equal to the number of unique materials
n = len(hog_vals)

# Similarity matrix initialization
dissimilarity_matrix = np.empty((n, n))


for k in range(n):
    for m in range(n):

        # Construct similarity matrix based on the 2D Euclidian distance between HoG parameters
        dissimilarity_matrix[k, m] = np.sqrt(np.sum((hog_vals[k] - hog_vals[m]) ** 2))


############ MDS Analysis #####################################

# Number of dimension to project similarity matrix
desired_dimension = 3
# Resultant projection coordinates
mds = MDS(n_components = desired_dimension, metric = True, n_jobs = 3, dissimilarity = "precomputed")

mds_result = mds.fit_transform(dissimilarity_matrix)

np.save("mds_result", mds_result)