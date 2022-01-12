#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "solvers.cuh"
#include "solvers.cu"


int main(int arc, char* argv[])
{
    std::cout<<"\n** Starting Jacobi Solver **\n";
    const int num_resolutions = 10;
    const int resolution_gpu[num_resolutions] = {100, 200, 500, 1000, 2000, 3000, 5000, 10000, 15000, 20000};
    const int iterations = 500;    
    bool useGPU;
    int numBlocks = 1;
    int blockSize = 256;

    std::ofstream out("measurements.csv");
    out<<"Resolution,CPU,GPU,GPU-Speedup\n";
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
        float speedup = elapsed_cpu.count() / elapsed_gpu.count();
        
        std::cout<<"\tCPU : "<<elapsed_cpu.count()<<" microseconds"<<std::endl;
        std::cout<<"\tGPU : "<<elapsed_gpu.count()<<" microseconds"<<std::endl;
        std::cout<<"\tSpeedup (GPU) : "<<speedup<<std::endl;
        out<<resolution<<","<<elapsed_cpu.count()<<","<<elapsed_gpu.count()<<","<<speedup<<"\n";
    }

    out.close();
    return 0;
}