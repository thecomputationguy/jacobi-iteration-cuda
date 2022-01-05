#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "solvers.cuh"
#include "solvers.cu"


int main(int arc, char* argv[])
{
    std::cout<<"\n** Starting Jacobi Solver **\n";
    const int resolution_gpu[5] = {10, 100, 1000, 2000, 3000};
    const int iterations = 1000;
    const int num_resolutions = 5;
    bool useGPU;
    int numBlocks = 1;
    int blockSize = 256;

    for(int i = 0; i < num_resolutions; i++)
    {
        const int resolution = resolution_gpu[i];

        // GPU code runs in this block
        useGPU = true;
        jacobiSolverGPU<float> jacobiGPU(resolution, useGPU, numBlocks, blockSize);
        std::cout<<"\nResolution : "<<resolution<<std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for(int j = 0; j < iterations; j++)
        {
            auto result = jacobiGPU.solve();
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto elapsed_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start) / iterations;

        // CPU code runs in this block
        useGPU = false;
        jacobiSolverCPU<float> jacobiCPU(resolution, useGPU);
        start = std::chrono::high_resolution_clock::now();
        for(int j = 0; j < iterations; j++)
        {
            auto result = jacobiCPU.solve();
        }
        stop = std::chrono::high_resolution_clock::now();
        auto elapsed_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start) / iterations;
        
        std::cout<<"\tCPU : "<<elapsed_cpu.count()<<" microseconds"<<std::endl;
        std::cout<<"\tGPU : "<<elapsed_gpu.count()<<" microseconds"<<std::endl;
    }

    return 0;
}