from math import sqrt
import numpy as np
import line_profiler

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


# distance matrix cpu version
@profile
def distance_3(positions):
    positions_1D = np.ravel(positions)
    a = positions.shape[0]
    b = positions.shape[1]
    dmRM_CPU=np.zeros((a, a),dtype=np.float32,order='C')
    dmRM_CPU = np.ravel(dmRM_CPU)
    for i in range(a):
        for j in range(i+1,a):
            dist = 0
            for k in range(b):
                ab = positions_1D[i*b + k] - positions_1D[j*b + k]
                dist += ab*ab
            dmRM_CPU[i*a + j] = np.sqrt(dist)
    return dmRM_CPU

distance_3(parts_2)
