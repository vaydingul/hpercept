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


def fetch_instances(fn, adj_set):

    # Open .hdf5 file to read content
    file = h5py.File(fn, "r")
    adjectives = file[adj_set].keys()
    dir_set = []
    counter = 0

    for adjective in adjectives:
        materials = file[adj_set][adjective].keys()

        for material in materials:

            dir_set.append(("/".join([adj_set, adjective, material]), counter))

        counter += 1

    return dir_set


def open_instance(instance, fn):

    # Open .hdf5 file to read content
    file = h5py.File(fn, "r")

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


def preprocess_instance(X, y, fixed_length=150):

    X_out = []

    for x in [X]:

        x.electrode_0 = (x.electrode_0 - np.mean(x.electrode_0, axis=0)) / \
            np.std(x.electrode_0, axis=0)

        x.pac_0 = (x.pac_0 - np.mean(x.pac_0, axis=0)) / \
            np.std(x.pac_0, axis=0)

        x.pac_0 = np.mean(x.pac_0, axis=1)

        x.pdc_0 = (x.pdc_0 - np.mean(x.pdc_0, axis=0)) / \
            np.std(x.pdc_0, axis=0)

        x.tac_0 = (x.tac_0 - np.mean(x.tac_0, axis=0)) / \
            np.std(x.tac_0, axis=0)

        x.tdc_0 = (x.tdc_0 - np.mean(x.tdc_0, axis=0)) / \
            np.std(x.tdc_0, axis=0)

        pca = PCA(n_components=4)
        x.electrode_0 = pca.fit_transform(x.electrode_0)

        # M = fit(PCA, x.electrode_0, maxoutdim=4, pratio=1.0)
        # x.electrode_0 = transform(M, x.electrode_0)

        x.electrode_1 = (x.electrode_1 - np.mean(x.electrode_1, axis=0)) / \
            np.std(x.electrode_1, axis=0)

        x.pac_1 = (x.pac_1 - np.mean(x.pac_1, axis=0)) / \
            np.std(x.pac_1, axis=0)

        x.pac_1 = np.mean(x.pac_1, axis=1)

        x.pdc_1 = (x.pdc_1 - np.mean(x.pdc_1, axis=0)) / \
            np.std(x.pdc_1, axis=0)

        x.tac_1 = (x.tac_1 - np.mean(x.tac_1, axis=0)) / \
            np.std(x.tac_1, axis=0)

        x.tdc_1 = (x.tdc_1 - np.mean(x.tdc_1, axis=0)) / \
            np.std(x.tdc_1, axis=0)

        pca = PCA(n_components=4)
        x.electrode_1 = pca.fit_transform(x.electrode_1)

        hold_ixs = x.controller_detail_state == EP_DICT["hold"]
        squeeze_ixs = x.controller_detail_state == EP_DICT["squeeze"]
        slow_slide_ixs = x.controller_detail_state == EP_DICT["slow_slide"]
        fast_slide_ixs = x.controller_detail_state == EP_DICT["fast_slide"]

        img = np.vstack(
            [
                resample(x.pac_0[hold_ixs], fixed_length),
                resample(x.pdc_0[hold_ixs], fixed_length),
                resample(x.tac_0[hold_ixs], fixed_length),
                resample(x.tdc_0[hold_ixs], fixed_length),
                resample(x.electrode_0[hold_ixs, :], fixed_length).T,
                resample(x.pac_1[hold_ixs], fixed_length),
                resample(x.pdc_1[hold_ixs], fixed_length),
                resample(x.tac_1[hold_ixs], fixed_length),
                resample(x.tdc_1[hold_ixs], fixed_length),
                resample(x.electrode_1[hold_ixs, :], fixed_length).T,
                resample(x.pac_0[squeeze_ixs], fixed_length),
                resample(x.pdc_0[squeeze_ixs], fixed_length),
                resample(x.tac_0[squeeze_ixs], fixed_length),
                resample(x.tdc_0[squeeze_ixs], fixed_length),
                resample(x.electrode_0[squeeze_ixs, :], fixed_length).T,
                resample(x.pac_1[squeeze_ixs], fixed_length),
                resample(x.pdc_1[squeeze_ixs], fixed_length),
                resample(x.tac_1[squeeze_ixs], fixed_length),
                resample(x.tdc_1[squeeze_ixs], fixed_length),
                resample(x.electrode_1[squeeze_ixs, :], fixed_length).T,
                resample(x.pac_0[slow_slide_ixs], fixed_length),
                resample(x.pdc_0[slow_slide_ixs], fixed_length),
                resample(x.tac_0[slow_slide_ixs], fixed_length),
                resample(x.tdc_0[slow_slide_ixs], fixed_length),
                resample(x.electrode_0[slow_slide_ixs, :], fixed_length).T,
                resample(x.pac_1[slow_slide_ixs], fixed_length),
                resample(x.pdc_1[slow_slide_ixs], fixed_length),
                resample(x.tac_1[slow_slide_ixs], fixed_length),
                resample(x.tdc_1[slow_slide_ixs], fixed_length),
                resample(x.electrode_1[slow_slide_ixs, :], fixed_length).T,
                resample(x.pac_0[fast_slide_ixs], fixed_length),
                resample(x.pdc_0[fast_slide_ixs], fixed_length),
                resample(x.tac_0[fast_slide_ixs], fixed_length),
                resample(x.tdc_0[fast_slide_ixs], fixed_length),
                resample(x.electrode_0[fast_slide_ixs, :], fixed_length).T,
                resample(x.pac_1[fast_slide_ixs], fixed_length),
                resample(x.pdc_1[fast_slide_ixs], fixed_length),
                resample(x.tac_1[fast_slide_ixs], fixed_length),
                resample(x.tdc_1[fast_slide_ixs], fixed_length),
                resample(x.electrode_1[fast_slide_ixs, :], fixed_length).T,
            ]
        )


        if np.sum(np.isnan(img)) != 0:

            img[np.isnan(img)] = np.mean(img[not np.isnan(img)])

        x.image = img

    return X, y
