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

    def __call__(self, X_train, y_train, X_test = None, y_test = None):
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

    def evaluate(self, desc_text=None, fn=None, noprint = True):

        evaluation_results = np.array([metric(self.y_test, self.model.predict(
            self.X_test)) for metric in self.metrics], dtype=np.ndarray)

        msg = ""

        msg += "Description: {}\n".format(desc_text)
        msg += "Coefficients: {}\n".format(
                    str(self.model.coef_))
        
        msg += "Intercept: {}\n".format(
                    str(self.model.intercept_))

        for (ix, metric) in enumerate(self.metrics):

            msg += "{}: {}\n".format(metric.__name__.replace("_", " ").title(),
                                                   str(evaluation_results[ix]))
        if not noprint:

            if fn is not None:

                with open(fn, "w") as f:

                    f.writelines(msg)
            
            else:

                print(msg)
        
        return msg



    def predict(self, sample):

        return self.model.predict(sample)

    def visualize(self, fn, desc=None):
        """

        visualize:

        Input:
            fn = Filename to save the figure
            desc = Descriptive text

        """
       
        for [X, y, name] in zip([self.X_train, self.X_test], [self.y_train, self.y_test], ["train", "test"]):
            
            # Initialize figure
            fig = plt.figure(figsize=(15.0, 15.0), dpi=200)

            # Method is different for 2D and 3D
        
            

            ax = fig.add_subplot(111, projection='3d')

            for k in range(X.shape[0]):

                marker = "+" if y[k] == 1 else "$-$" 
                color = "green" if self.predict([X[k]])[0] == y[k] else "red"
                data = X[k]

                ax.scatter(data[0], data[1], data[2],
                            color=color,
                            marker=marker)



            plt.title(self.evaluate(desc))

            plt.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.90])

            plt.savefig(fn.replace(".png","_{}.png".format(name)))

            plt.close("all")

       
