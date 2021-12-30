#include <cuda.h>
#include "../include/jacobi_gpu.cuh"

__global__
void jacobiGPUBasic(float* x_new, float* A, float* x_current, float* b, const int Nx, const int Ny)
{
    float sum = 0.0;
    int idx = threadIdx.x;
    int j;
    for(j = 0; j < Ny; j++)
    {
        if(idx != j)
        {
            sum += A[idx * Ny + j] * x_current[j];
        }
        x_new[idx] = (b[idx] - sum) / A[idx * Ny + idx];
    }
}