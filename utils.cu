#ifndef UTILS_CU
#define UTILS_CU

#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "utils.cuh"

/*
    Definition of methods in the hostCUDAVariable class.
*/

template <typename T>
hostCUDAVariable<T>::hostCUDAVariable(const size_t size, const bool useGPU) : size_(size), useGPU_(useGPU)
{
    /*
        Constructor : Allocation of memory on host and device, as required.
    */

    x_ = (T*)malloc(size_ * sizeof(T));

    if(useGPU_)
    {
        assert(cudaSuccess == cudaMalloc((void**) &xd_, size_ * sizeof(T)));
    }
    
}
       
template <typename T>
void hostCUDAVariable<T>::copyToDevice()
{
    /*
        Transfer of data from host to device.
    */

    assert(cudaSuccess == cudaMemcpy(xd_, x_, size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void hostCUDAVariable<T>::copyToHost()
{
    /*
        Transfer of data from device to host.
    */

    assert(cudaSuccess == cudaMemcpy(x_, xd_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
T*&  hostCUDAVariable<T>::getDeviceVariable()
{
    /*
        Fetch the variable from device.
    */

    return xd_;
}

template <typename T>
T*&  hostCUDAVariable<T>::getHostVariable()
{
    /*
        Fetch of data from host.
    */

    return x_;
}

template <typename T>
hostCUDAVariable<T>::~hostCUDAVariable()
{
    /*
        Destructor : De-allocation of memory on host and device, as required.
    */
    if(useGPU_) 
    {
        cudaFree(xd_);
    }              

    free(x_);
}

/*
    Definition of methods in the Solver class.
*/

template<typename T>
Solver<T>::Solver(const size_t size, const bool useGPU) : A_(size * size, useGPU), b_(size, useGPU), 
                                                        x_current_(size, useGPU), x_next_(size, useGPU), 
                                                        resolution_(size)
{
    /*
        Constructor : Get solver parameters and data structures.
    */
}

template<typename T>
T*& Solver<T>::solve()
{
    /*
        Pure Virtual Function : To be implemented in derived classes.
    */
}

std::vector<int> read_file()
{
    /*
        Method to read resolutions.
    */

    std::string filename = "resolutions.txt";
    std::vector<int> resolutions;
    std::string size;
    std::fstream file(filename, std::ios::in);

    if(file.is_open())
    {
        while(std::getline(file, size))
        {
            resolutions.push_back(stoi(size));
        }
    }

    return resolutions;
}

#endif