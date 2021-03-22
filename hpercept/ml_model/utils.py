import numpy as np
from numpy.lib.arraysetops import unique


def get_unique_adjectives(adjective_list):

	adjective_list = np.hstack([*adjective_list])
	unique_adjective_list = np.unique(adjective_list)
	return unique_adjective_list


def extract_adjective_encoding(adjective_list):

	unique_adjective_list = get_unique_adjectives(adjective_list)
	adjective_encoding = np.vstack([[1 if unique_adj in adj_list else 0 for unique_adj in unique_adjective_list] for adj_list in adjective_list])

	return adjective_encoding, unique_adjective_list

