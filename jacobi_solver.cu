#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "jacobi_CPU.hpp"
#include "jacobi_GPU.cuh"
#include "utils.cuh"
#include "utils.cu"

template<typename T>
class jacobiSolverGPU : public Solver<T>
{
    public:
        jacobiSolverGPU(size_t resolution, const bool useGPU) : Solver<T>(resolution, useGPU)
        {   
        }

        T*& solve()
        {
            auto x_next_device = Solver<T>::x_next_.getDeviceVariable();
            auto x_current_device = Solver<T>::x_current_.getDeviceVariable();
            auto b_device = Solver<T>::b_.getDeviceVariable();
            auto A_device = Solver<T>::A_.getDeviceVariable();
            const int numBlocks = 1;
            const int blockSize = 256;
            const size_t resolution = Solver<T>::resolution_;

            jacobiGPUBasic<<<numBlocks, blockSize>>>(x_next_device, A_device, x_current_device, b_device, resolution, resolution);
            Solver<T>::x_current_.copyToHost();

            return Solver<T>::x_current_.getHostVariable();
        }
};

template<typename T>
class jacobiSolverCPU : public Solver<T>
{
    public:
        jacobiSolverCPU(size_t resolution, const bool useGPU) : Solver<T>(resolution, useGPU)
        {   
        }

        T*& solve()
        {
            auto x_next = Solver<T>::x_next_.getHostVariable();
            auto x_current = Solver<T>::x_current_.getHostVariable();
            auto b = Solver<T>::b_.getHostVariable();
            auto A = Solver<T>::A_.getHostVariable();
            const size_t resolution = Solver<T>::resolution_;

            jacobiCPU(x_next, A, x_current, b, resolution, resolution);

            return Solver<T>::x_current_.getHostVariable();
        }
};


int main(int arc, char* argv[])
{
    std::cout<<"\n** Starting Jacobi Solver **\n";
    const int resolution_gpu[5] = {10, 100, 1000, 2000, 3000};
    const int iterations = 1000;
    const int num_resolutions = 5;
    bool useGPU;

    for(int i = 0; i < num_resolutions; i++)
    {
        const int resolution = resolution_gpu[i];

        useGPU = true;
        jacobiSolverGPU<float> jacobiGPU(resolution, useGPU);
        std::cout<<"\nResolution : "<<resolution<<std::endl;

        //std::cout<<"\nGPU calculation started."<<std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for(int j = 0; j < iterations; j++)
        {
            auto result = jacobiGPU.solve();
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto elapsed_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start) / iterations;
        //std::cout<<"GPU calculation done."<<std::endl;

        //std::cout<<"\nCPU calculation started."<<std::endl;
        useGPU = false;
        jacobiSolverCPU<float> jacobiCPU(resolution, useGPU);
        start = std::chrono::high_resolution_clock::now();
        for(int j = 0; j < iterations; j++)
        {
            auto result = jacobiCPU.solve();
        }
        stop = std::chrono::high_resolution_clock::now();
        auto elapsed_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start) / iterations;
        //std::cout<<"CPU calculation done."<<std::endl;

        
        std::cout<<"\tCPU : "<<elapsed_cpu.count()<<" microseconds"<<std::endl;
        std::cout<<"\tGPU : "<<elapsed_gpu.count()<<" microseconds"<<std::endl;
    }

    return 0;
}