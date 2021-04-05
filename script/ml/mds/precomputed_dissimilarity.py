#! NOT WORKING
from sklearn.manifold import MDS
from skimage.metrics import structural_similarity as ssim
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(1, "./")
sys.path.insert(2, "./../")

from hpercept.ml_model.utils import mse
from hpercept.ml_model import manifold


def create_dissimilarity_matrix(images, method):

	n = images.shape[0]

	dissimilarity_matrix = np.empty((n, n))

	for k in range(n):
		for m in range(n):

			dissimilarity_matrix[k, m] = method(images[k], images[m])

	return dissimilarity_matrix


if __name__ == "__main__":

	# Load presaved entities
	npz_loader = np.load("./entity/imgs_adjs_names.npz", allow_pickle=True)
	# Fetch images
	imgs = npz_loader["arr_0"]
	# Fetch adjectives
	adjs = npz_loader["arr_1"]
	# Fetch names
	names = npz_loader["arr_2"]

	method = MDS
	model_args = {"n_components": 3, "metric": True,
		"n_jobs": 4, "dissimilarity": "precomputed"}
	scalar_methods = [mse, ssim]

	for (ix, scalar_method) in enumerate(tqdm(scalar_methods)):

		dm = create_dissimilarity_matrix(imgs, scalar_method)


		model = manifold.ManifoldModelExecutor(method=MDS, method_args=model_args)
		model(dm)
		model.visualize("./{0}.png".format(ix), mean=False)
	   
