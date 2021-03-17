import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE


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

    def visualize(self, fn, group_count=10, normalize=True, mean=False):

        dim = self.model_args["n_components"]

        if normalize:

            self.result = (self.result - np.min(self.result, axis=0)) / \
                (np.max(self.result, axis=0) - np.min(self.result, axis=0))

        if mean:

            data = np.empty(
                (int(self.result.shape[0]/group_count), self.result.shape[1]))

            for (ix, k) in enumerate(range(0, self.result.shape[0], group_count)):

                data[ix] = np.mean(
                    self.result[k:(k+group_count), :], axis=0)

            group_count = 1

        else:

            data = self.result

        if self.method == MDS:

            eval_val = self.model.stress_

        elif self.method == TSNE:

            eval_val = self.model.kl_divergence_

        fig = plt.figure()

        if dim == 2:

            for k in range(data.shape[0]):

                plt.scatter(data[k:k+group_count, 0], data[k:k+group_count, 1])

            plt.title("Evaluation Value = {0}\n{1}\n{2}".format(eval_val, "\n".join([str(k) + " = " + str(v) for (
                k, v) in self.model_args.items()]), "\n".join([str(k) + " = " + str(v) for (k, v) in self.feature_extractor_args.items()])))
            plt.tight_layout()
            plt.savefig(fn)

        if dim == 3:

            ax = fig.add_subplot(111, projection='3d')
            for k in range(data.shape[0]):

                ax.scatter(data[k:k+group_count, 0], data[k:k +
                                                          group_count, 1], data[k:k+group_count, 2])

            plt.title("Evaluation Value = {0}\n{1}\n{2}".format(eval_val, "\n".join([str(k) + " = " + str(v) for (
                k, v) in self.model_args.items()]), "\n".join([str(k) + " = " + str(v) for (k, v) in self.feature_extractor_args.items()])))
            plt.tight_layout()
            plt.savefig(fn)

        plt.close("all")
