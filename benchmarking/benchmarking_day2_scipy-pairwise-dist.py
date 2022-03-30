from scipy.spatial import distance_matrix
import line_profiler
import numpy as np

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
parts_100000 = np.random.rand(np.int32(1e6), 3)

@profile
def test_pairwise_Scipy(positions):
    for particle_id in range(positions.shape[0]):
        a = distance_matrix(positions)
        print(a)


test_pairwise_Scipy(parts_2)