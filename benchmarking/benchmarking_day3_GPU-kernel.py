import scipy
import line_profiler
import numpy as np


@profile
def CPU_Mani(XRM):
    nparticle = XRM.shape[0]
    ndimensions = XRM.shape[1]
    XRM = np.ravel(XRM)
    dmRM_CPU = np.zeros((nparticle, nparticle), dtype = np.float32, order='C')
    dmRM_CPU = np.ravel(dmRM_CPU)
    for i in range(nparticle-1):
        for j in range(i+1, nparticle-1):
            dist = 0
            for k in range(ndimensions):
                ab = XRM[i*ndimensions + k] - XRM[j*ndimensions + k]
                dist += ab*ab
            dmRM_CPU[i*ndimensions + j] = np.sqrt(dist)


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
def test_CPU_pairwise_Mani(positions):
    CPU_Mani(positions)


test_CPU_pairwise_Mani(parts_10)