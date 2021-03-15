import h5py
import numpy as np
from torch._C import StringType
from . import phac
from sklearn.decomposition import PCA
from scipy.signal import resample

EP_DICT = {"hold": b'HOLD_FOR_10_SECONDS',
           "squeeze": b'SQUEEZE_SET_PRESSURE_SLOW',
           "slow_slide": b'SLIDE_5CM',
           "fast_slide": b'MOVE_DOWN_5CM'}


def fetch_instances(file, adj_set):

    
    adjectives = file[adj_set].keys()
    dir_set = []
    

    for adjective in adjectives:
        materials = file[adj_set][adjective].keys()

        for material in materials:

            dir_set.append("/".join([adj_set, adjective, material]))

    return np.array(dir_set)


def open_instance(instance, file):

    

    X = phac.PHAC2(
        np.array(file[instance]["accelerometer"]),
        np.array(file[instance]["biotacs/finger_0/electrodes"]),
        np.array(file[instance]["biotacs/finger_0/pac"]),
        np.array(file[instance]["biotacs/finger_0/pdc"]),
        np.array(file[instance]["biotacs/finger_0/tac"]),
        np.array(file[instance]["biotacs/finger_0/tdc"]),
        np.array(file[instance]["biotacs/finger_1/electrodes"]),
        np.array(file[instance]["biotacs/finger_1/pac"]),
        np.array(file[instance]["biotacs/finger_1/pdc"]),
        np.array(file[instance]["biotacs/finger_1/tac"]),
        np.array(file[instance]["biotacs/finger_1/tdc"]),
        np.array(file[instance]["state/controller_detail_state"]),
        None,
        np.array(file[instance]["adjectives"]),
    )

    return X


def preprocess_instance(X, fixed_length=150):

    X_out = []



    X.electrode_0 = (X.electrode_0 - np.mean(X.electrode_0, axis=0)) / \
        np.std(X.electrode_0, axis=0)

    X.pac_0 = (X.pac_0 - np.mean(X.pac_0, axis=0)) / \
        np.std(X.pac_0, axis=0)

    X.pac_0 = np.mean(X.pac_0, axis=1)

    X.pdc_0 = (X.pdc_0 - np.mean(X.pdc_0, axis=0)) / \
        np.std(X.pdc_0, axis=0)

    X.tac_0 = (X.tac_0 - np.mean(X.tac_0, axis=0)) / \
        np.std(X.tac_0, axis=0)

    X.tdc_0 = (X.tdc_0 - np.mean(X.tdc_0, axis=0)) / \
        np.std(X.tdc_0, axis=0)

    pca = PCA(n_components=4)
    X.electrode_0 = pca.fit_transform(X.electrode_0)

    X.electrode_1 = (X.electrode_1 - np.mean(X.electrode_1, axis=0)) / \
        np.std(X.electrode_1, axis=0)

    X.pac_1 = (X.pac_1 - np.mean(X.pac_1, axis=0)) / \
        np.std(X.pac_1, axis=0)

    X.pac_1 = np.mean(X.pac_1, axis=1)

    X.pdc_1 = (X.pdc_1 - np.mean(X.pdc_1, axis=0)) / \
        np.std(X.pdc_1, axis=0)

    X.tac_1 = (X.tac_1 - np.mean(X.tac_1, axis=0)) / \
        np.std(X.tac_1, axis=0)

    X.tdc_1 = (X.tdc_1 - np.mean(X.tdc_1, axis=0)) / \
        np.std(X.tdc_1, axis=0)

    pca = PCA(n_components=4)
    X.electrode_1 = pca.fit_transform(X.electrode_1)

    hold_ixs = X.controller_detail_state == EP_DICT["hold"]
    squeeze_ixs = X.controller_detail_state == EP_DICT["squeeze"]
    slow_slide_ixs = X.controller_detail_state == EP_DICT["slow_slide"]
    fast_slide_ixs = X.controller_detail_state == EP_DICT["fast_slide"]

    img = np.vstack(
        [
            resample(X.pac_0[hold_ixs], fixed_length),
            resample(X.pdc_0[hold_ixs], fixed_length),
            resample(X.tac_0[hold_ixs], fixed_length),
            resample(X.tdc_0[hold_ixs], fixed_length),
            resample(X.electrode_0[hold_ixs, :], fixed_length).T,
            resample(X.pac_1[hold_ixs], fixed_length),
            resample(X.pdc_1[hold_ixs], fixed_length),
            resample(X.tac_1[hold_ixs], fixed_length),
            resample(X.tdc_1[hold_ixs], fixed_length),
            resample(X.electrode_1[hold_ixs, :], fixed_length).T,
            resample(X.pac_0[squeeze_ixs], fixed_length),
            resample(X.pdc_0[squeeze_ixs], fixed_length),
            resample(X.tac_0[squeeze_ixs], fixed_length),
            resample(X.tdc_0[squeeze_ixs], fixed_length),
            resample(X.electrode_0[squeeze_ixs, :], fixed_length).T,
            resample(X.pac_1[squeeze_ixs], fixed_length),
            resample(X.pdc_1[squeeze_ixs], fixed_length),
            resample(X.tac_1[squeeze_ixs], fixed_length),
            resample(X.tdc_1[squeeze_ixs], fixed_length),
            resample(X.electrode_1[squeeze_ixs, :], fixed_length).T,
            resample(X.pac_0[slow_slide_ixs], fixed_length),
            resample(X.pdc_0[slow_slide_ixs], fixed_length),
            resample(X.tac_0[slow_slide_ixs], fixed_length),
            resample(X.tdc_0[slow_slide_ixs], fixed_length),
            resample(X.electrode_0[slow_slide_ixs, :], fixed_length).T,
            resample(X.pac_1[slow_slide_ixs], fixed_length),
            resample(X.pdc_1[slow_slide_ixs], fixed_length),
            resample(X.tac_1[slow_slide_ixs], fixed_length),
            resample(X.tdc_1[slow_slide_ixs], fixed_length),
            resample(X.electrode_1[slow_slide_ixs, :], fixed_length).T,
            resample(X.pac_0[fast_slide_ixs], fixed_length),
            resample(X.pdc_0[fast_slide_ixs], fixed_length),
            resample(X.tac_0[fast_slide_ixs], fixed_length),
            resample(X.tdc_0[fast_slide_ixs], fixed_length),
            resample(X.electrode_0[fast_slide_ixs, :], fixed_length).T,
            resample(X.pac_1[fast_slide_ixs], fixed_length),
            resample(X.pdc_1[fast_slide_ixs], fixed_length),
            resample(X.tac_1[fast_slide_ixs], fixed_length),
            resample(X.tdc_1[fast_slide_ixs], fixed_length),
            resample(X.electrode_1[fast_slide_ixs, :], fixed_length).T,
        ]
    )


    if np.sum(np.isnan(img)) != 0:

        img[np.isnan(img)] = np.mean(img[not np.isnan(img)])

    X.image = img

    return X
