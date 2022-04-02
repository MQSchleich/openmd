#!/usr/bin/env python
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from math import sqrt
import numpy as np

kerneltemplate = SourceModule(
    """
#include <math.h>
#include <stdio.h>
// not sure where the file is
//#include <common/helper_math.h>

inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ double dot(double3 a, double3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double3 operator*(double b, double3 a)
{
    return make_double3(b * a.x, b * a.y, b * a.z);
}


__global__ void
gpuForcesParallelo(double3 *forces, double3 *positions, double epsilon, double sig, int N){
    
    const int idx = threadIdx.x+blockIdx.x*blockDim.x;
    
    double3 dvec;
    double distance, distance_sq, distance_six, distance_eight, prefactor, bracket, result;
    double sigma_six = sig*sig*sig*sig*sig*sig;
    double prefactor_lj = 24.0*epsilon*sigma_six;
    
    if(idx < N) 
    {
        for(int j = 0; j < N; j++)
        {
            if(j == idx)
                continue;

            dvec = positions[j] - positions[idx];

            distance = dot(dvec,dvec);
            distance_sq = 1.0/distance;
            distance_six = distance_sq*distance_sq*distance_sq;
            distance_eight = distance_six*distance_sq;
            
            prefactor = prefactor_lj*distance_eight;
            bracket = ((2.0*sigma_six)*distance_six - 1.0);
            result = prefactor*bracket;
            
            forces[idx] += result * dvec;
                
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

    X = np.array([[0,0,0], [1,0,0]], dtype=np.float64,order='C')
    # X = np.array([[0,0,0], [1,0,0], [3,-2,0], [11,-2,3]], dtype=np.float64,order='C')
    # X = np.array([[0, 1, 0], [5, 5, 5], [6, 0, 6], [4,3,5]],dtype=np.float32,order='C')
    XRM = np.ravel(X)
    ndim = np.shape(X)[0]
    nvdim = np.shape(X)[1]


    forces = np.ravel(np.zeros((ndim, 3), dtype=np.float64))

    forces_gpu = gpuarray.empty(shape=forces.shape, dtype=np.float64)

    XRM_gpu = gpuarray.to_gpu(XRM)


    f_gpu = kerneltemplate.get_function("gpuForcesParallelo")
    # f_gpu = kerneltemplate.get_function("gpuForcesC")
    f_gpu(
        forces_gpu,
        XRM_gpu,
        np.float64(1),
        np.float64(1),
        np.int32(ndim),
        block=(ndim, 1, 1),
        stream=stream[0],
        grid=(1, 1, 1)
    )

    stream[0].synchronize()

    print(np.reshape(forces_gpu.get(),(ndim,3), order='C'))