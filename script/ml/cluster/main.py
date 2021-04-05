from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from skimage.feature import hog
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.ml_model import cluster, manifold


n_components_options = [3, 2]
metrics = [False, True]
orientations_options = [5, 6, 7, 8]
pixels_per_cell_options = [(16, 16), (24, 24), (32, 32)]
cells_per_block_options = [(1, 1), (2, 2)]




def get_embeddings(imgs, method, feature_extractor, model_args, feature_extractor_args):

	mme = manifold.ManifoldModelExecutor(
		method, feature_extractor, model_args, feature_extractor_args)

	return mme(imgs)

def model_creator():

	models = []
	for k in range(1,25,3):
	# Configuration for clustering algorithm
		method = AgglomerativeClustering
		feature_extractor = hog
		model_args = {"n_clusters":k,"linkage":"ward"}
		feature_extractor_args = {"orientations": 5, "pixels_per_cell": (
			32, 32), "cells_per_block": (2, 2)}

		models.append(cluster.ClusterExecutor(method, feature_extractor, model_args, feature_extractor_args))

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

	# Configuration for embedding algorithm
	method = MDS
	feature_extractor = hog
	model_args = {"n_components": 3, "metric": True, "n_jobs": 4}
	feature_extractor_args = {"orientations": 5, "pixels_per_cell": (
		32, 32), "cells_per_block": (2, 2)}

	embeddings = get_embeddings(
		imgs, method, feature_extractor, model_args, feature_extractor_args)

	models = model_creator()

	for (ix, model) in enumerate(tqdm(models)):

		model(imgs)
		model.visualize(
			"./entity/cluster_images/{0}.png".format(ix), embeddings)
	
