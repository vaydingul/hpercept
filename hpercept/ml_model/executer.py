import numpy as np

class ModelExecutor:
	"""

	"""

	def __init__(self, method, feature_extractor = None, **args):

		self.method = method
		self.args = args
		self.feature_extractor = feature_extractor

	def __call__(self, data):

		n = data.shape[0]
		data_ = np.empty((n,n))

		for k in range(n):
			for m in range(n):

				data_[k,m] =self.feature_extractor(data[k], data[m])				

		return self.method(**self.args).fit_transform(data)