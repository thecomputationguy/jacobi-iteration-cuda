#include "jacobi_CPU.hpp"
#include <omp.h>

void jacobiCPU(float* x_new, float* A, float* x_current, float* b, const int Nx, const int Ny)
{
    int i, j;
    float sum;
    const int THREADS = 4;

    #pragma omp parallel for num_threads(THREADS) private(i,j)
    for(i = 0; i < Nx; i++)
    {
        sum = 0;
        for(j = 0; j < Ny; j++)
        {
            if(i != j)
                sum += A[i * Ny + j] * x_current[j];
        }
        x_new[i] = (b[i] - sum) / A[i * Ny + i];
    }
}