import scipy
import line_profiler
import numpy as np


@profile
def calc_pairwise_distance(particle, positions, box_length):
    """Standard euclidean pairwise distance. Is abstracted to enable innovation here.

    Args:
        particle (_type_): 
        positions (_type_): 3D positions
        box_length (_type_): box_length
    """
    particle = positions[particle, :]
    difference = positions - particle
    difference[difference > box_length / 2] -= box_length
    difference[difference <= -box_length / 2] += box_length
    difference = np.linalg.norm(difference, axis=0)
    return difference


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
def test_pairwise_ZA(positions):
    for particle_id in range(positions.shape[0]):
        calc_pairwise_distance(particle_id, positions, box_length)


test_pairwise_ZA(parts_100000)