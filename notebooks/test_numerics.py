#!/usr/bin/env python
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np

kerneltemplate = SourceModule(
    """
#include <math.h>
#include <stdio.h>
__global__ void
gpuForcesParallelo(double *f, double *positions, double epsilon, double sig, int N, int dim){
    
    const int idx = threadIdx.x+blockIdx.x*blockDim.x;
    
    double distance, distance_sq, distance_six, distance_eight, prefactor, bracket, result;
    double sigma_six = sig*sig*sig*sig*sig*sig;
    double prefactor_lj = 24.0*epsilon*sigma_six;
    double dx, dy, dz;
    
    if(idx < N) {
        for(int j = idx + 1; j < N; j++){
            
            dx = positions[j*3]   - positions[idx*3];
            dy = positions[j*3+1] - positions[idx*3+1];
            dz = positions[j*3+2] - positions[idx*3+2]; 

            distance = dx*dx + dy*dy + dz*dz;
            distance_sq = 1.0/distance;
            distance_six = distance_sq*distance_sq*distance_sq;
            distance_eight = distance_six*distance_sq;

            prefactor = prefactor_lj*distance_eight;
            bracket = ((2.0*sigma_six)*distance_six - 1.0);
            result = prefactor*bracket;
            
            printf("result %f \\n", double(result));

            if (dim == 0) {
                f[idx] += result * dx;
            }
            if (dim == 1) {
                f[idx] += result * dy;
            }
            if (dim == 2) {
                f[idx] += result * dz;
            }
        }
    }
    __syncthreads();
}


"""
)

for i in range(1):
    print("run: ", i)

    stream = []
    stream.append(drv.Stream())
    stream.append(drv.Stream())
    stream.append(drv.Stream())

    # X = np.array([[0,0,0], [1,0,0]], dtype=np.float64,order='C')
    X = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=np.float64, order="C")
    # X = np.array([[0,0,0], [1,0,0], [3,-2,0], [11,-2,3]], dtype=np.float64,order='C')
    # X = np.array([[0, 1, 0], [5, 5, 5], [6, 0, 6], [4,3,5]],dtype=np.float32,order='C')
    XRM = np.ravel(X)
    ndim = np.shape(X)[0]
    nvdim = np.shape(X)[1]

    fx = np.ravel(np.zeros((ndim, 1), dtype=np.float64))
    fy = np.ravel(np.zeros((ndim, 1), dtype=np.float64))
    fz = np.ravel(np.zeros((ndim, 1), dtype=np.float64))

    fx_gpu = gpuarray.empty(shape=fx.shape, dtype=np.float64)
    fy_gpu = gpuarray.empty(shape=fy.shape, dtype=np.float64)
    fz_gpu = gpuarray.empty(shape=fz.shape, dtype=np.float64)

    XRM_gpu = gpuarray.to_gpu(XRM)

    f_gpu = kerneltemplate.get_function("gpuForcesParallelo")
    # gpuForcesParallelo(double *fx, double *positions, double epsilon, double sig, int N){
    f_gpu(
        fz_gpu,
        XRM_gpu,
        np.float64(1),
        np.float64(1),
        np.int32(ndim),
        np.int32(2),
        block=(16, 1, 1),
        stream=stream[0],
        grid=(1, 1, 1),
    )
    """
    f_gpu(
        fy_gpu,
        XRM_gpu,
        np.float64(1),
        np.float64(1),
        np.int32(ndim),
        np.int32(1),
        block=(16, 1, 1),
        stream=stream[1],
        grid=(1, 1, 1)
    )
    f_gpu(
        fz_gpu,
        XRM_gpu,
        np.float64(1),
        np.float64(1),
        np.int32(ndim),
        np.int32(2),
        block=(16, 1, 1),
        stream=stream[2],
        grid=(1, 1, 1)
    )"""

    stream[0].synchronize()

    # print(fx_gpu.get())
    # print(fy_gpu.get())
    print(fz_gpu.get())
