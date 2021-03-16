from torch.utils import data
from sklearn.manifold import MDS
from skimage.feature import hog
import sys
import numpy as np

sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.ml_model import executer

n_components_options = [3, 2]
metrics = [False, True]
orientations_options = [5, 6, 7, 8]
pixels_per_cell_options = [(16, 16), (24, 24), (32, 32)]
cells_per_block_options = [(1, 1), (2, 2)]


def benchmark():

    models = []

    for n_components in n_components_options:
        for metric in metrics:
            for orientations in orientations_options:
                for pixels_per_cell in pixels_per_cell_options:
                    for cells_per_block in cells_per_block_options:

                        cfg_model = {"n_components": n_components,
                                     "metric": metric,
                                     "n_jobs": 4}

                        cfg_feature_extractor = {"orientations": orientations,
                                                 "pixels_per_cell": pixels_per_cell,
                                                 "cells_per_block": cells_per_block}
                        


                        models.append(executer.ModelExecutor(
                            MDS, hog, cfg_model, cfg_feature_extractor))

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

    models = benchmark()
    results = []
    for model in models:

        results.append(model(imgs))
        print(model.model.stress_)
