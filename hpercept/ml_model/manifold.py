import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import feature
from sklearn.manifold import MDS, TSNE
from matplotlib import rcParams
from distinctipy import distinctipy

metric_dict = {MDS: "stress_", TSNE: "kl_divergence_"}


class ManifoldModelExecutor:
    """

    ManifoldModelExecutor

    It is a general manifold learning execution class which is located on the top of the sklearn
    """

    def __init__(self, method, feature_extractor=None, method_args=None, feature_extractor_args=None):
        """

        Constructor:

        Input:
            model = ML model 
            feature_extractor = Feature extractor function 
            method_args = ML model arguments
            feature_extractor_args = Feature extractor function arguments

        """
        # Model and its constructor parameters
        self.method = method
        self.method_args = method_args
        # Feature extractor function its arguments
        self.feature_extractor_args = feature_extractor_args
        self.feature_extractor = feature_extractor
        # Model initialization
        self.model = self.method(**self.method_args)

        # Some initilialization
        self.data = None
        self.data_extracted = None
        self.result = None

    def __call__(self, data):
        """

        Call method:

        Input:
            data = The data to be processed by model and feature extractor

        Output:
            self.result = The processed data

        """

        self.data_extracted = data
        # Get number of samples
        n = self.data_extracted.shape[0]

        if self.feature_extractor is not None and self.feature_extractor_args is not None:
            # Apply feature extractor to the each sample in the data
            self.data_extracted = np.array([self.feature_extractor(
                self.data_extracted[k], **self.feature_extractor_args) for k in range(n)])

        # Apply model to the extracted data
        self.result = self.model.fit_transform(self.data_extracted)
        # Return the result
        return self.result

    def visualize(self, fn, group_count=10, normalize=True, mean=False):
        """

        visualize:

        Input:
            fn = Filename to save the figure
            group_count = Each ??group_count?? element of the self.result will be grouped
            normalize = If it is True, then, the all data will be normalized
            mean = If it is True, then, the ach group will be replaced by its mean


        """
        # Get the set dimension
        dim = self.method_args["n_components"]

        if normalize:
            # Apply naive normalization
            self.result = (self.result - np.min(self.result, axis=0)) / \
                (np.max(self.result, axis=0) - np.min(self.result, axis=0))

        data = self.result

        if mean:

            # If mean is True, then cluster the data based on the group_count

            data = np.empty(
                (int(self.result.shape[0]/group_count), self.result.shape[1]))

            for (ix, k) in enumerate(range(0, self.result.shape[0], group_count)):

                data[ix] = np.mean(
                    self.result[k:(k+group_count), :], axis=0)

            group_count = 1

        # Get metric value based on the method
        eval_val = self.get_evaluation()#self.model.__dict__[metric_dict[self.method]]

        # Initialize figure
        fig = plt.figure(figsize=(15.0, 15.0), dpi=200)

        # Create visually distinct color space for ease of visualization
        #color_interval = np.linspace(
        #    0, 1, int(0.8 * data.shape[0]/group_count))
        #colors = [cm.prism(x) for x in color_interval]
        colors = distinctipy.get_colors(int(data.shape[0]/group_count))
        markers = ["o", "^", "s", "H", "X"]

        ss = [rcParams['lines.markersize']**3, 2 * rcParams['lines.markersize']
              ** 3, 0.5 * rcParams['lines.markersize']**3]
        
        
        
        # Method is different for 2D and 3D

        if dim == 2:

            for (ix, k) in enumerate(range(0, data.shape[0], group_count)):

                plt.scatter(data[k:k+group_count, 0],
                            data[k:k+group_count, 1],
                            color=colors[ix],
                            marker=str(np.random.choice(markers, 1)[0]),
                            s=np.random.choice(ss, 1)[0])

        if dim == 3:

            ax = fig.add_subplot(111, projection='3d')

            for (ix, k) in enumerate(range(0, data.shape[0], group_count)):

                ax.scatter(data[k:k+group_count, 0],
                           data[k:k + group_count, 1], data[k:k+group_count, 2],
                           color=colors[ix],
                           marker=str(np.random.choice(markers, 1)[0]),
                           s=np.random.choice(ss, 1)[0])

        plt.title("Evaluation Value = {0}\n{1}\n{2}".format(eval_val, " || ".join([str(k) + " = " + str(v) for (
            k, v) in self.method_args.items()]), " || ".join([str(k) + " = " + str(v) for (k, v) in self.feature_extractor_args.items()])))

        plt.tight_layout()#(pad=0.1, rect=[0, 0.03, 1, 0.90])
        
        plt.savefig(fn)

        plt.close("all")


    def get_evaluation(self):
        """

        It returns the fitness score which is 
        defined for the specified method/model.

        """
        return self.model.__dict__[metric_dict[self.method]]