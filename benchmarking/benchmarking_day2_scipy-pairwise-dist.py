from scipy.spatial import distance
import line_profiler
import numpy as np

# np.random.seed(0)

box_length = 6  # shape

# 2 particles: particle, positions
parts_2 = np.random.rand(2, 3)

# 10 particles
parts_10 = np.random.rand(10, 3)

# 100 particles
parts_100 = np.random.rand(100, 3)

# 1'000 particles
parts_1000 = np.random.rand(1000, 3)

# 10'000 particles
parts_10000 = np.random.rand(10000, 3)

# 100'000 particles
parts_100000 = np.random.rand(np.int32(1e5), 3)


@profile
def test_pairwise_Scipy(positions):
    distance.pdist(positions)  # = root(x^2 + y^2 + z^2)


test_pairwise_Scipy(parts_100000)