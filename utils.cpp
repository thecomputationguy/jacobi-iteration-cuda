#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include "utils.hpp"

template <typename T>
hostCUDAVariable<T>::hostCUDAVariable(const size_t size, const bool useGPU) : size_(size), useGPU_(useGPU)
{
    x_ = (T*)malloc(size_ * sizeof(T));

    if(useGPU_)
    {
        assert(cudaSuccess == cudaMalloc((void**) &xd_, size_ * sizeof(T)));
    }
    
}
       
template <typename T>
void hostCUDAVariable<T>::copyToDevice()
{
    assert(cudaSuccess == cudaMemcpy(xd_, x_, size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void hostCUDAVariable<T>::copyToHost()
{
    assert(cudaSuccess == cudaMemcpy(x_, xd_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
T*&  hostCUDAVariable<T>::getDeviceVariable()
{
    return xd_;
}

template <typename T>
T*&  hostCUDAVariable<T>::getHostVariable()
{
    return x_;
}

template <typename T>
hostCUDAVariable<T>::~hostCUDAVariable()
{
    if(useGPU_) 
    {
        cudaFree(xd_);
    }              

    free(x_);
}