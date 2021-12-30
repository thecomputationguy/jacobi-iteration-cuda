#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <assert.h>
#include <iostream>
// #include "include/jacobi_cpu.h"
// #include "include/jacobi_gpu.cuh"

// void jacobiCPU(float* x_new, const float* A, float* x_current, float* b, const int Nx, const int Ny, const int iterations)
// {
//     int i, j;
//     float sum;

//     for(i = 0; i < Nx; i++)
//     {
//         sum = 0.0;
//         for(j = 0; j < Ny; j++)
//         {
//             if(i != j)
//                 sum += A[i * Ny + j] * x_current[j];
//         }
//         x_new[i] = (b[i] - sum) / A[i * Ny + i];
//     }
// }

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

void allocateHostMemory(float **x_current, float **x_next, float **b, float **A, int resolution)
{
    printf("\nAllocating Host Memory...");
    *x_current = (float*)malloc(resolution * sizeof(float));
    *x_next = (float*)malloc(resolution * sizeof(float));
    *b = (float*)malloc(resolution * sizeof(float));
    *A = (float*)malloc(resolution * resolution * sizeof(float));
    printf("\nHost Memory Allocated.\n");
}

void allocateDeviceMemory(float **x_current_device, float **x_next_device, float **b_device, float **A_device, 
                        int resolution,
                        float **x_current, float **x_next, float **b, float **A)
{
    printf("\nAllocating Device Memory...");
    assert(cudaSuccess == cudaMalloc((void**) x_current_device, resolution * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void**) x_next_device, resolution * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void**) b_device, resolution * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void**) A_device, resolution * resolution * sizeof(float) * sizeof(float)));
    printf("\nDevice Memory Allocated.\n");

    printf("\nCopying to Device Memory...");
    assert(cudaSuccess == cudaMemcpy(*x_current_device, *x_current, resolution * sizeof(float), cudaMemcpyHostToDevice));
    assert(cudaSuccess == cudaMemcpy(*x_next_device, *x_next, resolution * sizeof(float), cudaMemcpyHostToDevice));
    assert(cudaSuccess == cudaMemcpy(*b_device, *b, resolution * sizeof(float), cudaMemcpyHostToDevice));
    assert(cudaSuccess == cudaMemcpy(*A_device, *A, resolution * resolution * sizeof(float), cudaMemcpyHostToDevice));
    printf("\nCopied to Device.");
}

void freeDeviceMemory(float **x_current_device, float **x_next_device, float **b_device, float **A_device)
{
    assert(cudaSuccess == cudaFree(*A_device));
    assert(cudaSuccess == cudaFree(*x_current_device));
    assert(cudaSuccess == cudaFree(*x_next_device));
    assert(cudaSuccess == cudaFree(*b_device));
}

void freeHostMemory(float **x_current, float **x_next, float **b, float **A)
{
    free(*A);
    free(*x_current);
    free(*x_next);
    free(*b);
}

int main(int arc, char* argv[])
{
    unsigned int resolution = 10000;
    unsigned int increment = 10000;
    unsigned int final_resolution = 40000;
    unsigned int iterations = 50;
    clock_t start_time;
    clock_t end_time;
    double elapsed_time;
    int blockSize = ceil(resolution / 768);
    int numBlocks = 1;

    printf("\n** Starting Jacobi Solver on CPU **\n");
    // while(resolution <= final_resolution)
    // {
    //     float* x_current = (float*)malloc(resolution * sizeof(float));
    //     float* x_next = (float*)malloc(resolution * sizeof(float));
    //     float* b = (float*)malloc(resolution * sizeof(float));
    //     float* A = (float*)malloc(resolution * resolution * sizeof(float));

    //     start_time = clock();        
    //     jacobiCPU(x_next, A, x_current, b, resolution, resolution);
    //     end_time = clock();
    //     elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;

    //     printf("\nResolution       : %d", resolution);
    //     printf("\nIterations       : %d", iterations);
    //     printf("\nTime Elapsed (s) : %.2lf", elapsed_time / iterations);
    //     printf("\n");

    //     free(x_current);
    //     free(x_next);
    //     free(b);
    //     free(A);

    //     resolution += increment;
    // }

    printf("\n** Starting Jacobi Solver on GPU (Basic) **\n");
    const int resolution_gpu[5] = {10, 100, 1000, 10000, 15000};
    iterations = 1000;
    std::cout<<"\n trying C++"<<std::endl;

    for (int iter = 0; iter < 5; iter++)
    {
        const int resolution = resolution_gpu[iter];
        
        float *x_current_device, *x_next_device, *b_device, *A_device;
        float *x_current, *x_next, *b, *A;
        allocateHostMemory(&x_current, &x_next, &b, &A, resolution);

        allocateDeviceMemory(&x_current_device, &x_next_device, &b_device, &A_device,
                            resolution,
                            &x_current, &x_next, &b, &A);

        start_time = clock();
        for(int i = 0; i < iterations; i++)
        {
            jacobiGPUBasic<<<numBlocks, blockSize>>>(x_next_device, A_device, x_current_device, b_device, resolution, resolution);
        }
        end_time = clock();
        elapsed_time = (end_time - start_time) ;

        printf("\nResolution       : %d", resolution);
        printf("\nIterations       : %d", iterations);
        printf("\nTime Elapsed (s) : %lf", elapsed_time);
        printf("\n");

        freeDeviceMemory(&x_current_device, &x_next_device, &b_device, &A_device);
        freeHostMemory(&x_current, &x_next, &b, &A);        
    }

    return 0;
}