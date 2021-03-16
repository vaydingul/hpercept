import numpy as np
import matplotlib.pyplot as plt


class ModelExecutor:
    """

    """

    def __init__(self, method, feature_extractor=None, model_args=None, feature_extractor_args=None):

        self.method = method
        self.model_args = model_args
        self.feature_extractor_args = feature_extractor_args
        self.feature_extractor = feature_extractor
        self.model = self.method(**self.model_args)
        self.data = None
        self.data_extracted = None
        self.result = None

    def __call__(self, data):

        self.data = data
        n = self.data.shape[0]

        self.data_extracted = []

        for k in range(n):

            self.data_extracted.append(self.feature_extractor(
                self.data[k], **self.feature_extractor_args))

        self.data_extracted = np.array(self.data_extracted)
        self.result = self.model.fit_transform(self.data_extracted)

        return self.result

    def visualize(self, fn, dim, grouped=True, group_count=10, normalize=True, mean=False):

        if normalize:

            self.result = (self.result - np.min(self.result, axis=0)) / \
                (np.max(self.result, axis=0) - np.min(self.result, axis=0))

        if (grouped == False) and (mean == True):

            partial = np.empty((self.result.shape[0]/group_count, ))

            for (ix, k) in enumerate(range(0, self.result.shape[0], group_count)):

                partial[ix] = np.mean(
                    self.result[k:(k+group_count), :], axis=0)

        elif (grouped == True) and (mean == False):

            partial = self.result

			
