import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mds_result = np.load("mds_result.npy")

fig = plt.figure()
ax = fig.gca(projection='3d')


ax.scatter(mds_result[:, 0], mds_result[:, 1])#, mds_result[:, 2])
plt.show()
