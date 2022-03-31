#!/usr/bin/env python
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from math import sqrt
import numpy as np

# get device information
MAX_THREADS_PER_BLOCK = drv.Device(0).get_attribute(
    pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK
)
BLOCK_SIZE = int(sqrt(MAX_THREADS_PER_BLOCK))
print(MAX_THREADS_PER_BLOCK, BLOCK_SIZE)

kerneltemplate = SourceModule(
    """
#include <math.h>
#include <stdio.h>
__global__ void
gpuForcesC(float *fx, float *fy, float *fz, float *positions, float epsilon, float sig, int n_dim){
    float distance, distance_sq, distance_six, distance_eight, prefactor, bracket, result;
    float sigma_six = sig*sig*sig*sig*sig*sig;
    float prefactor_lj = 24.0*epsilon*sigma_six;
    float dx, dy, dz;
    for(int i = 0; i<n_dim-1; i++){
        for(int j = i+1; j<n_dim; j++){
            dx = positions[j*3] -   positions[i*3];
            dy = positions[j*3+1] - positions[i*3+1];
            dz = positions[j*3+2] - positions[i*3+2];
            distance = sqrtf(dx*dx + dy*dy + dz*dz);
            distance = 1/distance;
            distance_sq = distance*distance;
            distance_six = distance_sq*distance_sq*distance_sq;
            distance_eight = distance_six*distance_sq;
            prefactor = prefactor_lj*distance_eight;
            bracket = ((2*sigma_six)*distance_six - 1);

            result = prefactor*bracket;
            fx[i] += result * dx;
            fy[i] += result * dy;
            fz[i] += result * dz;
            fx[j] -= result * dx;
            fy[j] -= result * dy;
            fz[j] -= result * dz;
            
        }

        //printf("pr %lf %lf %f %f %f %f %f %f\\n",double(distance), double(distance_sq), double(distance_six), double(distance_eight), float(prefactor_lj), float(dx), float(dy), float(dz));
        //printf("pr %lf %lf %f %f %f\\n",double(positions[3]), double(result), double(fx[i]), double(fy[i]), double(fz[i]));
    }
    
}

"""
)

stream = []
stream.append(drv.Stream())

X = np.random.random((100000, 3))
# X = np.array([[0, 1, 0], [5, 5, 5], [6, 0, 6], [4,3,5]],dtype=np.float32,order='C')
XRM = np.ravel(X)
ndim = np.shape(X)[0]
nvdim = np.shape(X)[1]


fx = np.ravel(np.zeros((ndim, 1), dtype=np.float32))
fy = np.ravel(np.zeros((ndim, 1), dtype=np.float32))
fz = np.ravel(np.zeros((ndim, 1), dtype=np.float32))

fx_gpu = gpuarray.empty(shape=fx.shape, dtype=np.float32)
fy_gpu = gpuarray.empty(shape=fy.shape, dtype=np.float32)
fz_gpu = gpuarray.empty(shape=fz.shape, dtype=np.float32)

XRM_gpu = gpuarray.to_gpu(XRM)


f_gpu = kerneltemplate.get_function("gpuForcesC")
f_gpu(
    fx_gpu,
    fy_gpu,
    fz_gpu,
    XRM_gpu,
    np.float32(1),
    np.float32(1),
    np.int32(ndim),
    block=(1, 1, 1),
    stream=stream[0],
    grid=(1, 1, 1),
)

stream[0].synchronize()
