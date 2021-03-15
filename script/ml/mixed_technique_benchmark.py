from torch.utils import data
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from skimage.feature import hog, local_binary_pattern


methods = {MDS: ["n_components", "metric", "n_jobs"], TSNE: [
    "n_components", "perplexity", "early_exaggeration", "learning_rate", "n_jobs"]}
feature_extractors = {hog: ["orientations",
                            "pixels_per_cell", "cells_per_block", ]}


def benchmark():
    for (method, opts) in methods.items():
        for (feature_extractor, opts_) in feature_extractors.items():

            pass


if __name__ == "__main__":

    benchmark()
