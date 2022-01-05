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
    __syncthreads();
}
template <typename T>
class hostCUDAVariable
{
    private:
        /* data */
        T* x_ ;
        T* xd_ ;
        const size_t size_;
    
    public:
       
        hostCUDAVariable(const size_t size):size_(size)
        {
            x_ = (T*)malloc(size_ * sizeof(T));
            //std::cout<<"\nAllocated Memory for Host."<<std::endl;
            
            assert(cudaSuccess == cudaMalloc((void**) &xd_, size_ * sizeof(T)));
            //std::cout<<"\nAllocated Memory for Device."<<std::endl;
        }

        void copyToDevice()
        {
            assert(cudaSuccess == cudaMemcpy(xd_, x_, size_ * sizeof(T), cudaMemcpyHostToDevice));
            //std::cout<<"\nCopied to Device."<<std::endl;
        }

        void copyToHost()
        {
            assert(cudaSuccess == cudaMemcpy(x_, xd_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
            //std::cout<<"\nCopied to Host."<<std::endl;
        }

        T*& getDeviceVariable()
        {
            return xd_;
        }

        T*& getHostVariable()
        {
            return x_;
        }

        ~hostCUDAVariable()
        {
            cudaFree(xd_);
            //std::cout<<"\nDeallocated Memory for Device."<<std::endl;
            free(x_);
            //std::cout<<"\nDeallocated Memory for Host."<<std::endl;
        }
};

template<typename T>
class Solver
{
    protected:
        hostCUDAVariable<T> A_, b_, x_current_, x_next_;
        const size_t resolution_;

    public:
        Solver(const size_t size) : A_(size * size), b_(size), x_current_(size), x_next_(size), resolution_(size)
        {
            std::cout<<"\nConstructor called for Solver"<<std::endl;
        }

        virtual T*& solve()
        {

        }
};

template<typename T>
class jacobiSolver : public Solver<T>
{
    private:
        size_t resolution_;

    public:
        jacobiSolver(size_t resolution) : Solver<T>(resolution)
        {   
            jacobiSolver::resolution_ = resolution;
            std::cout<<"\nConstructor called for Jacobi"<<std::endl;
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
            std::cout<<"\nGPU Calculation done."<<std::endl;
            Solver<T>::x_current_.copyToHost();

            return Solver<T>::x_current_.getHostVariable();
        }
};


int main(int arc, char* argv[])
{
    unsigned int resolution = 10000;
    unsigned int iterations = 50;
    clock_t start_time;
    clock_t end_time;
    double elapsed_time;

    std::cout<<"\n** Starting Jacobi Solver on CPU **\n"<<std::endl;

    printf("\n** Starting Jacobi Solver on GPU (Basic) **\n");
    const int resolution_gpu[5] = {10, 100, 1000, 10000, 15000};
    iterations = 1000;
    resolution = 5;

    jacobiSolver<float> jacobi(resolution);
    auto result = jacobi.solve();
    for(int i=0; i < resolution; i++)
        std::cout<<result[i]<<std::endl;

    return 0;
}