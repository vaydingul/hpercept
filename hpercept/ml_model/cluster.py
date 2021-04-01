import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import feature
from sklearn.cluster import *
from matplotlib import rcParams


class ClusterExecutor:
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
        self.model.fit(self.data_extracted)
        # Return the result
        

    def visualize(self, fn, data):
        """

        visualize:

        Input:
            fn = Filename to save the figure
            data = The data to be visualized in a clustered manner


        """

        # Initialize figure
        fig = plt.figure(figsize=(15.0, 15.0), dpi=300)

        # Create visually distinct color space for ease of visualization
        color_interval = np.linspace(
            0, 1, int(self.model.n_clusters_))
        colors = [cm.prism(x) for x in color_interval]

        markers = ["o", "^", "s", "H", "X"]

        ss = [rcParams['lines.markersize']**3, 2 * rcParams['lines.markersize']
              ** 3, 0.5 * rcParams['lines.markersize']**3]

        ax = fig.add_subplot(111, projection='3d')

        for k in range(self.model.n_clusters_):

            data_ = data[self.model.labels_ == k]

            ax.scatter(data_[:, 0],
                       data_[:, 1],
                       data_[:, 2],
                       color=colors[np.random.choice(len(colors))],
                       marker=str(np.random.choice(markers, 1)[0]),
                       s=np.random.choice(ss, 1)[0])

        plt.title("Number of clusters = {0}\n{1}".format(self.model.n_clusters_, " || ".join([str(k) + " = " + str(v) for (
            k, v) in self.method_args.items()])))

        plt.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.90])

        plt.savefig(fn)
        plt.close("all")

    def dendogram(self):
        pass
    
