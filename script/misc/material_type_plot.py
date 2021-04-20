
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.manifold import MDS
from skimage.feature import hog
from distinctipy import distinctipy

sys.path.insert(1, "./")
sys.path.insert(2, "./../")

from hpercept.data_handle import utils, phac
from hpercept.ml_model import utils
from hpercept.ml_model import manifold

def get_embeddings(imgs, method, feature_extractor, model_args, feature_extractor_args):

	mme = manifold.ManifoldModelExecutor(
		method, feature_extractor, model_args, feature_extractor_args)

	return mme(imgs)


def visualize(embedding, names):

	names_material_dict = {name: phac.MATERIAL_TYPE_DICT["_".join(
		name.split("_")[:-2])] for name in names}

	# Initialize figure
	fig = plt.figure(figsize=(15.0, 15.0), dpi=300)

	unique_material_type_number = len(phac.UNIQUE_MATERIAL_TYPES)
	colors = distinctipy.get_colors(unique_material_type_number)

	markers = ["o", "^", "s", "H", "X"]

	ss = [rcParams['lines.markersize'] ** 3,
		  2 * rcParams['lines.markersize'] ** 3,
		  0.5 * rcParams['lines.markersize'] ** 3]

	ax = fig.add_subplot(111, projection='3d')

	for (ix, material_type) in enumerate(phac.UNIQUE_MATERIAL_TYPES):
		
		data_ = embedding[[names_material_dict[name] == material_type for name in names]]

		ax.scatter(data_[:, 0],
				   data_[:, 1],
				   data_[:, 2],
				   color=colors[ix],
				   marker=str(np.random.choice(markers, 1)[0]),
				   s=np.random.choice(ss, 1)[0],
				   label = material_type.title())

	plt.title("Embedding based on material types")

	plt.tight_layout()#(pad=0.1, rect=[0, 0.03, 1, 0.90])
	plt.legend(loc = "best")
	plt.savefig("deneme.png")
	#plt.close("all")
	plt.show()


if __name__ == "__main__":

	# Load presaved entities
	npz_loader = np.load(
		"./entity/imgs_imgs_normalized_adjs_names.npz", allow_pickle=True)
	# Fetch images
	imgs = npz_loader["arr_0"]
	# Fetch images
	imgs_normalized = npz_loader["arr_1"]
	# Fetch adjectives
	adjs = npz_loader["arr_2"]
	# Fetch names
	names = npz_loader["arr_3"]

	# Adjective encoding
	adjective_encoding, unique_adjective_list = utils.extract_adjective_encoding(
		adjs)

	# Configuration for embedding algorithm
	method = MDS
	feature_extractor = hog
	model_args = {"n_components": 3, "metric": True, "n_jobs": 4}
	feature_extractor_args = {"orientations": 5, "pixels_per_cell": (
		32, 32), "cells_per_block": (2, 2)}

	embeddings = get_embeddings(
		imgs, method, feature_extractor, model_args, feature_extractor_args)

	visualize(embeddings, names)