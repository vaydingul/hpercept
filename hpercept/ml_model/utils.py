import numpy as np
from numpy.lib.arraysetops import unique
import random


def get_unique_adjectives(adjective_list):
    """

    From all of the adjective lists, it returns the total
    unique adjective list.

    """

    # Place each list tip-to-tip
    adjective_list = np.hstack([*adjective_list])
    # Fetch unique set
    unique_adjective_list = np.unique(adjective_list)

    return unique_adjective_list


def extract_adjective_encoding(adjective_list):
    """

    It returns the binary encoding of the whole adjective list.

    If there is m unique adjectives, and n samples,
    then the returned matrix will have the size of (n x m).

    """

    # Fetch unique adjectives
    unique_adjective_list = get_unique_adjectives(adjective_list)
    # Binary encoding for each sample in the dataset
    adjective_encoding = np.vstack(
        [[1 if unique_adj in adj_list else 0 for unique_adj in unique_adjective_list] for adj_list in adjective_list])

    return adjective_encoding, unique_adjective_list


def unit_vector(vector):
    """

            Returns the unit vector of the vector.

            """

    # Divide vector with its norm
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """

            Returns the angle in radians between vectors 'v1' and 'v2'

    """
    # Trasnform into unit vectors
    v1_u = unit_vector(v1.ravel())
    v2_u = unit_vector(v2.ravel())

    # Then, from the geometry, calculate the angle between vectors
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_colors(n):
    """

    Create n visually distinct colors.

    """
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
		
        r += step
        g += step
        b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r/256, g/256, b/256, 1.0))
    return ret


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype(np.float) - imageB.astype(np.float)) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err