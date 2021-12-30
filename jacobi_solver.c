#include<stdio.h>
#include<stdlib.h>
#include<getopt.h>
#include<time.h>
#include "jacobi_cpu.h"
#include "utils.h"

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

    float* x_current = (float*)malloc(Nx * sizeof(float));
    float* x_next = (float*)malloc(Nx * sizeof(float));
    float* b = (float*)malloc(Nx * sizeof(float));
    float* A = (float*)malloc(Nx * Ny * sizeof(float));

    printf("\n**Starting Jacobi Solver on CPU**\n");
    while(resolution <= final_resolution)
    {
        float* x_current = (float*)malloc(resolution * sizeof(float));
        float* x_next = (float*)malloc(resolution * sizeof(float));
        float* b = (float*)malloc(resolution * sizeof(float));
        float* A = (float*)malloc(resolution * resolution * sizeof(float));

        start_time = clock();
        for(int i = 0; i < iterations; i++)
        {
            jacobiCPU(x_next, A, x_current, b, resolution, resolution);
        }
        end_time = clock();
        elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;

        printf("\nResolution       : %d", resolution);
        printf("\nTime Elapsed (s) : %f", elapsed_time);
        printf("\n");

        resolution += increment;
    }

    return 0;
}