
import sys
import numpy as np
from numpy.lib.arraysetops import unique
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(1, "./")
sys.path.insert(2, "./../")

from hpercept.ml_model import regressor, manifold
from hpercept.ml_model import utils




def get_embeddings(imgs, method, feature_extractor, model_args, feature_extractor_args):

    mme = manifold.ManifoldModelExecutor(
        method, feature_extractor, model_args, feature_extractor_args)

    return mme(imgs)


if __name__ == "__main__":

    # Load presaved entities
    npz_loader = np.load("./entity/imgs_adjs_names.npz", allow_pickle=True)
    # Fetch images
    imgs = npz_loader["arr_0"]
    # Fetch adjectives
    adjs = npz_loader["arr_1"]
    # Fetch names
    names = npz_loader["arr_2"]

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

    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5)

    train_index, test_index =  list(sss.split(embeddings, adjective_encoding))[0]
    embeddings_train, embeddings_test = embeddings[train_index], embeddings[test_index]
    adjectives_train, adjectives_test = adjective_encoding[train_index], adjective_encoding[test_index]

    """
    embeddings_train, embeddings_test, adjectives_train, adjectives_test = train_test_split(
        embeddings, adjective_encoding, test_size=0.5, shuffle=True, stratify = True)
    """

    scaler = StandardScaler()
    embeddings_train = scaler.fit_transform(embeddings_train)
    embeddings_test = scaler.transform(embeddings_test)

    lr_models_per_adjective = [regressor.RegressionModelExecutor(method = LogisticRegression, 
        method_args = {}, metrics = [accuracy_score, confusion_matrix]) for _ in range(unique_adjective_list.shape[0])]

    for (ix, lr_model) in enumerate(lr_models_per_adjective):
        
        lr_model(embeddings_train, adjectives_train[:, ix],embeddings_test, adjectives_test[:, ix])

        #lr_model.evaluate(unique_adjective_list[ix])

        lr_model.visualize("./entity/lr_images/{}.png".format(ix), unique_adjective_list[ix])


    """
    angles = np.empty((len(lr_models_per_adjective),
                       len(lr_models_per_adjective)))

    for (ix1, m1) in enumerate(lr_models_per_adjective):
        for (ix2, m2) in enumerate(lr_models_per_adjective):

            angles[ix1, ix2] = utils.angle_between(m1.coef_, m2.coef_)

    import pandas as pd

    df = pd.DataFrame(angles*180/np.pi)
    df.to_excel("angles.xlsx")
    """
    print("Done!")
