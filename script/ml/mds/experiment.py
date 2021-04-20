from sklearn.manifold import MDS
from sklearn.model_selection import ParameterGrid
from skimage.feature import hog
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.ml_model import manifold

method_opts = {"n_components": [3, 2],
			   "metric": [True, False],
			   "n_jobs": [-1]}

feature_extractor_opts = {"orientations": [5, 6, 7, 8],
						  "pixels_per_cell": [(16, 16), (24, 24), (32, 32)],
						  "cells_per_block": [(1, 1), (2, 2)]}


def model_creator():

	models = []

	for method_opt in ParameterGrid(method_opts):
		for feature_extractor_opt in ParameterGrid(feature_extractor_opts):

			models.append(manifold.ManifoldModelExecutor(
				MDS, hog, method_opt, feature_extractor_opt))

	return models


if __name__ == "__main__":

	# Load presaved entities
	npz_loader = np.load("./entity/imgs_imgs_normalized_adjs_names.npz", allow_pickle=True)
	# Fetch images
	imgs = npz_loader["arr_0"]
	# Fetch images
	imgs_normalized = npz_loader["arr_1"]
	# Fetch adjectives
	adjs = npz_loader["arr_2"]
	# Fetch names
	names = npz_loader["arr_3"]

	models = model_creator()
	evaluation_values = []

	for (ix, model) in enumerate(tqdm(models)):

		model(imgs)
		model.visualize("./entity/mds_images/hog/normal/{0}.png".format(ix), mean = False)
		model.visualize("./entity/mds_images/hog/clustered/{0}.png".format(ix), mean = True)
		evaluation_values.append(model.get_evaluation())

	min_ix = np.argmin(evaluation_values)
	
	models[min_ix].visualize(
		"./entity/mds_images/hog/normal/fittest.png", mean=False)
	models[min_ix].visualize(
		"./entity/mds_images/hog/clustered/fittest.png", mean=True)

		
		

