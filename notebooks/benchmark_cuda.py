import time

import numpy as np
from numba import njit


@njit(parallel=True)
def force_pre_cuda(positions, constants, box_length):
    num = positions.shape[0]
    positions = np.ravel(positions)
    epsilon = constants[0]
    sigma = constants[1]
    sigma_six = sigma**6
    prefactor_lj = 24 * epsilon * sigma_six
    fx = np.zeros(num)
    fy = np.zeros(num)
    fz = np.zeros(num)
    for i in range(0, num - 1):
        for j in range(i + 1, num):
            ###############
            dx = positions[j * 3] - positions[i * 3]
            dy = positions[j * 3 + 1] - positions[i * 3 + 1]
            dz = positions[j * 3 + 2] - positions[i * 3 + 2]
            distance = dx * dx + dy * dy + dz * dz
            distance_sq = 1 / distance
            distance_sq = distance * distance
            distance_six = distance_sq * distance_sq * distance_sq
            distance_eight = distance_six * distance_sq
            prefactor = prefactor_lj * distance_eight
            bracket = (2.0 * sigma_six) * distance_six - 1.0
            result = prefactor * bracket
            fx[i] += result * dx
            fy[i] += result * dy
            fz[i] += result * dz
            fx[j] -= result * dx
            fy[j] -= result * dy
            fz[j] -= result * dz
    return fx, fy, fz


num_rep = 10
positions = np.array([[1, 1, 1], [1, 2, 2]])
force_pre_cuda(positions, np.array([1.0, 1.0]), 10)
for particles in [10, 100, 1000, 10000, 100000]:
    if particles == 100000:
        num_rep = 10
    positions = np.random.random((particles, 3))
    print(len(positions), "particles")
    times = np.zeros(num_rep)
    for i in range(num_rep):
        a = time.time()
        force_pre_cuda(positions, np.array([1.0, 1.0]), 1000)
        b = time.time()
        times[i] = b - a
    print("Mean:", np.mean(times), "Std: ", np.std(times))
