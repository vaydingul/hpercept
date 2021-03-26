import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import feature
from sklearn.manifold import MDS, TSNE
from matplotlib import rcParams

metric_dict = {MDS: "stress_", TSNE: "kl_divergence_"}


class RegressionModelExecutor:
    """

    RegressionModelExecutor

    It is a general regression analysis execution class which is located on the top of the sklearn
    """

    def __init__(self, method, method_args=None, metrics=None):
        """

        Constructor:

        Input:
            method = Regression type/model 
            feature_extractor = Feature extractor function 
            method_args = ML model arguments
            feature_extractor_args = Feature extractor function arguments

        """
        # Model and its constructor parameters
        self.method = method
        self.method_args = method_args
        self.metrics = metrics
        # Model initialization
        self.model = self.method(**self.method_args)

        # Some initilialization
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def __call__(self, X_train, y_train, X_test, y_test):
        """

        Call method:

        Input:
            X_train = Train feature matrix
            y_train = Train target vector
                        X_test = Test feature matrix
            y_test = Test target vector
        Output:

        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, desc_text=None, fn=None):

        evaluation_results = np.array([metric(self.y_test, self.model.predict(
            self.X_test)) for metric in self.metrics], dtype=np.ndarray)

        if fn is not None:

            with open("out.txt", "w") as f:

                f.writelines("Description: {}\n".format(desc_text))
                f.writelines("Coefficients: {}\n".format(
                    str(self.model.coef_)))
                f.writelines("Intercept: {}\n".format(
                    str(self.model.intercept_)))

                for (ix, metric) in enumerate(self.metrics):

                    f.writelines("{}: {}\n".format(metric.__name__.replace("_", " ").title(),
                                                   str(evaluation_results[ix])))
                    f.writelines("\n\n")

    def visualize(self, fn, group_count=10, normalize=True, mean=False):
        """

        visualize:

        Input:
            fn = Filename to save the figure
            group_count = Each ´group_count´ element of the self.result will be grouped
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
        eval_val = self.model.__dict__[metric_dict[self.method]]

        # Initialize figure
        fig = plt.figure(figsize=(15.0, 15.0), dpi=200)

        # Create visually distinct color space for ease of visualization
        color_interval = np.linspace(
            0, 1, int(0.8 * data.shape[0]/group_count))
        colors = [cm.prism(x) for x in color_interval]

        markers = ["o", "^", "s", "H", "X"]

        ss = [rcParams['lines.markersize']**3, 2 * rcParams['lines.markersize']
              ** 3, 0.5 * rcParams['lines.markersize']**3]

        # Method is different for 2D and 3D

        if dim == 2:

            for (ix, k) in enumerate(range(0, data.shape[0], group_count)):

                plt.scatter(data[k:k+group_count, 0],
                            data[k:k+group_count, 1],
                            color=colors[np.random.choice(len(colors))],
                            marker=str(np.random.choice(markers, 1)[0]),
                            s=np.random.choice(ss, 1)[0])

        if dim == 3:

            ax = fig.add_subplot(111, projection='3d')

            for (ix, k) in enumerate(range(0, data.shape[0], group_count)):

                ax.scatter(data[k:k+group_count, 0],
                           data[k:k + group_count, 1], data[k:k+group_count, 2],
                           color=colors[np.random.choice(len(colors))],
                           marker=str(np.random.choice(markers, 1)[0]),
                           s=np.random.choice(ss, 1)[0])

        plt.title("Evaluation Value = {0}\n{1}\n{2}".format(eval_val, " || ".join([str(k) + " = " + str(v) for (
            k, v) in self.method_args.items()]), " || ".join([str(k) + " = " + str(v) for (k, v) in self.feature_extractor_args.items()])))

        plt.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.90])

        plt.savefig(fn)

        plt.close("all")
