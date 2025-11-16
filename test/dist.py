import numpy as np
import ot

A = np.random.randn(50, 2)
B = np.random.rand(60, 2)
M = ot.dist(A, B)  # cost matrix
wasserstein_dist = ot.emd2([], [], M)  # EMD squared distance
print(wasserstein_dist)
