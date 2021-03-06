#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "solvers.cuh"
#include "solvers.cu"


int main(int arc, char* argv[])
{
    std::cout<<"\n** Starting Jacobi Solver **\n";
    std::vector<int> resolutions = read_file();
    const int iterations = 500;    
    bool useGPU;
    int numBlocks = 1;
    int blockSize = 256;
    std::chrono::steady_clock::time_point start; // start timer
    std::chrono::steady_clock::time_point stop; // stop timer

    std::ofstream out("measurements.csv");
    out<<"Resolution,CPU,GPU,GPU-Speedup\n";
    for(int i = 0; i < resolutions.size(); i++)
    {
        const int resolution = resolutions[i];

        // GPU code runs in this block
        useGPU = true;
        jacobiSolverGPU<float> jacobiGPU(resolution, useGPU, numBlocks, blockSize);
        std::cout<<"\nResolution : "<<resolution<<std::endl;
        start = std::chrono::steady_clock::now();
        for(int j = 0; j < iterations; j++)
        {
            auto result = jacobiGPU.solve();
        }
        stop = std::chrono::steady_clock::now();
        auto elapsed_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / iterations;

        // CPU code runs in this block
        useGPU = false;
        jacobiSolverCPU<float> jacobiCPU(resolution, useGPU);
        start = std::chrono::steady_clock::now();
        for(int j = 0; j < iterations; j++)
        {
            auto result = jacobiCPU.solve();
        }
        stop = std::chrono::steady_clock::now();
        auto elapsed_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / iterations;
        float speedup = elapsed_cpu / elapsed_gpu;
        
        std::cout<<"\tCPU : "<<elapsed_cpu<<" microseconds"<<std::endl;
        std::cout<<"\tGPU : "<<elapsed_gpu<<" microseconds"<<std::endl;
        std::cout<<"\tSpeedup (GPU) : "<<speedup<<std::endl;
        out<<resolution<<","<<elapsed_cpu<<","<<elapsed_gpu<<","<<speedup<<"\n";
    }

    out.close();
    return 0;
}