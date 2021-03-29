from sklearn.manifold import TSNE
from skimage.feature import hog
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.ml_model import manifold


n_components_options = [3, 2]
perplexity_options = [10, 15.0, 20.0, 25.0,  30.0]
learning_rate_options = [10.0, 50.0, 100.0]
orientations_options = [6, 7, 8]
pixels_per_cell_options = [(24, 24), (32, 32)]
cells_per_block_options = [(1, 1), (2, 2)]


def model_creator():

	models = []

	for n_components in n_components_options:
		for perplexity in perplexity_options:
			for learning_rate in learning_rate_options:
				for orientations in orientations_options:
					for pixels_per_cell in pixels_per_cell_options:
						for cells_per_block in cells_per_block_options:

							cfg_model = {"n_components": n_components,
										 "perplexity": perplexity,
										 "learning_rate": learning_rate,
										 "n_jobs": 4}

							cfg_feature_extractor = {"orientations": orientations,
													 "pixels_per_cell": pixels_per_cell,
													 "cells_per_block": cells_per_block}

							models.append(manifold.ManifoldModelExecutor(
								TSNE, hog, cfg_model, cfg_feature_extractor))

	return models


if __name__ == "__main__":

	# Load presaved entities
	npz_loader = np.load("./entity/imgs_adjs_names.npz", allow_pickle=True)
	# Fetch images
	imgs = npz_loader["arr_0"]
	# Fetch adjectives
	adjs = npz_loader["arr_1"]
	# Fetch names
	names = npz_loader["arr_2"]

	models = model_creator()
	evaluation_values = []

	for (ix, model) in enumerate(tqdm(models)):

		model(imgs)
		evaluation_values.append(model.get_evaluation())

	min_ix = np.argmin(evaluation_values)
	models[min_ix].visualize(
			"./entity/mds_images/normal/fittest.png", mean=False)
	models[min_ix].visualize(
			"./entity/mds_images/clustered/fittest.png", mean=True)