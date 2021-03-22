
import sys
import numpy as np
from numpy.lib.arraysetops import unique
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from skimage.feature import hog
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
sys.path.insert(1, "./")
sys.path.insert(2, "./../")

from hpercept.ml_model import executer
from hpercept.ml_model import utils

def logistic_regressor(embeddings, labels):

    lr = LogisticRegression()

    lr.fit(embeddings, labels)

    return lr


def get_embeddings(imgs, method, feature_extractor, model_args, feature_extractor_args):

    me = executer.ModelExecutor(
        method, feature_extractor, model_args, feature_extractor_args)

    return me(imgs)

def evaluate(model, x_test, y_test, *methods):

    evaluation_results = np.array([method(y_test, model.predict(x_test)) for method in methods], dtype = np.ndarray)

    return evaluation_results





if __name__ == "__main__":

    # Load presaved entities
    npz_loader = np.load("./entity/imgs_adjs_names.npz", allow_pickle=True)
    # Fetch images
    imgs = npz_loader["arr_0"]
    # Fetch adjectives
    adjs = npz_loader["arr_1"]
    # Fetch names
    names = npz_loader["arr_2"]

    adjective_encoding, unique_adjective_list = utils.extract_adjective_encoding(
        adjs)

    method = MDS
    feature_extractor = hog
    model_args = {"n_components": 3, "metric": True, "n_jobs": 4}
    feature_extractor_args = {"orientations":5, "pixels_per_cell": (32,32), "cells_per_block":(2,2)}

    embeddings = get_embeddings(imgs, method, feature_extractor, model_args, feature_extractor_args)


    embeddings_train, embeddings_test, adjectives_train,adjectives_test = train_test_split(embeddings, adjective_encoding, test_size=0.25)

    lr_models_per_adjective = [logistic_regressor(embeddings_train, adjectives_train[:, k]) for k in range(unique_adjective_list.shape[0])]

    evaluation_results = np.vstack([evaluate(lr_model, embeddings_test, adjectives_test[:, ix], confusion_matrix, accuracy_score) for (ix, lr_model) in enumerate(lr_models_per_adjective)])
    

    print("Done!")
