#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda.h>
#include "include/jacobi_cpu.h"
#include "include/jacobi_gpu.cuh"


int main(int arc, char* argv[])
{
    unsigned int Nx=4, Ny=4;
    unsigned int resolution = 10000;
    unsigned int increment = 1000;
    unsigned int final_resolution = 12000;
    unsigned int iterations = 20;
    clock_t start_time;
    clock_t end_time;
    double elapsed_time;
    int blockSize = ceil(resolution / 768);
    int numBlocks = 1;

    float *x_current_device, *x_next_device, *b_device, *A_device;

    float* x_current = (float*)malloc(resolution * sizeof(float));
    float* x_next = (float*)malloc(resolution * sizeof(float));
    float* b = (float*)malloc(resolution * sizeof(float));
    float* A = (float*)malloc(resolution * resolution * sizeof(float));

    cudaMalloc((void**) &x_current_device, resolution * sizeof(float));
    cudaMalloc((void**) &x_next_device, resolution * sizeof(float));
    cudaMalloc((void**) &b_device, resolution * sizeof(float));
    cudaMalloc((void**) &A_device, resolution * resolution * sizeof(float) * sizeof(float));

    cudaMemcpy(x_current_device, x_current, resolution * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_next_device, x_next, resolution * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b, resolution * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(A_device, A, resolution * resolution * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n** Starting Jacobi Solver on CPU **\n");
    while(resolution <= final_resolution)
    {
        start_time = clock();
        for(int i = 0; i < iterations; i++)
        {
            jacobiCPU(x_next, A, x_current, b, resolution, resolution);
        }
        end_time = clock();
        elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;

        printf("\nResolution       : %d", resolution);
        printf("\nIterations       : %d", iterations);
        printf("\nTime Elapsed (s) : %f", elapsed_time);
        printf("\n");

        resolution += increment;
    }

    printf("\n** Starting Jacobi Solver on GPU (Basic) **\n");
    while(resolution <= final_resolution)
    {
        start_time = clock();
        for(int i = 0; i < iterations; i++)
        {
            jacobiGPUBasic<<<numBlocks, blockSize>>>(x_next, A, x_current, b, resolution, resolution);
        }
        end_time = clock();
        elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;

        printf("\nResolution       : %d", resolution);
        printf("\nIterations       : %d", iterations);
        printf("\nTime Elapsed (s) : %f", elapsed_time);
        printf("\n");

        resolution += increment;
    }

    free(A);
    free(x_current);
    free(x_next);
    free(b);

    cudaFree(A_device);
    cudaFree(x_current_device);
    cudaFree(x_next_device);
    cudaFree(b_device);


    return 0;
}