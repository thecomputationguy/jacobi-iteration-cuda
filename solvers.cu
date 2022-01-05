#include <cuda.h>
#include "jacobi_CPU.hpp"
#include "jacobi_GPU.cuh"
#include "solvers.cuh"

template<typename T>
jacobiSolverGPU<T>::jacobiSolverGPU(size_t resolution, const bool useGPU, const int numBlocks, const int blockSize) : Solver<T>(resolution, useGPU), numBlocks_(numBlocks), blockSize_(blockSize)
{ 
}

template<typename T>
T*& jacobiSolverGPU<T>::solve()
{
    auto x_next_device = Solver<T>::x_next_.getDeviceVariable();
    auto x_current_device = Solver<T>::x_current_.getDeviceVariable();
    auto b_device = Solver<T>::b_.getDeviceVariable();
    auto A_device = Solver<T>::A_.getDeviceVariable();
    const size_t resolution = Solver<T>::resolution_;

    jacobiGPUBasic<<<numBlocks_, blockSize_>>>(x_next_device, A_device, x_current_device, b_device, resolution, resolution);
    Solver<T>::x_current_.copyToHost();

    return Solver<T>::x_current_.getHostVariable();
}

template<typename T>
jacobiSolverCPU<T>:: jacobiSolverCPU(size_t resolution, const bool useGPU) : Solver<T>(resolution, useGPU)
{   
}

template<typename T>
T*& jacobiSolverCPU<T>::solve()
{
    auto x_next = Solver<T>::x_next_.getHostVariable();
    auto x_current = Solver<T>::x_current_.getHostVariable();
    auto b = Solver<T>::b_.getHostVariable();
    auto A = Solver<T>::A_.getHostVariable();
    const size_t resolution = Solver<T>::resolution_;

    jacobiCPU(x_next, A, x_current, b, resolution, resolution);

    return Solver<T>::x_current_.getHostVariable();
}


