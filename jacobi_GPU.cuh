#ifndef JACOBI_GPU
#define JACOBI_GPU

__global__
void jacobiGPUBasic(float* x_new, float* A, float* x_current, float* b, const int Nx, const int Ny);

#endif