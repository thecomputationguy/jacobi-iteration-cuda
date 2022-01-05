#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <iostream>
#include <chrono>
// #include "include/jacobi_cpu.h"
// #include "include/jacobi_gpu.cuh"

void jacobiCPU(float* x_new, const float* A, float* x_current, float* b, const int Nx, const int Ny)
{
    int i, j;
    float sum;

    for(i = 0; i < Nx; i++)
    {
        sum = 0.0;
        for(j = 0; j < Ny; j++)
        {
            if(i != j)
                sum += A[i * Ny + j] * x_current[j];
        }
        x_new[i] = (b[i] - sum) / A[i * Ny + i];
    }
}

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
        bool useGPU_;
    
    public:
       
        hostCUDAVariable(const size_t size, const bool useGPU) : size_(size), useGPU_(useGPU)
        {
            x_ = (T*)malloc(size_ * sizeof(T));

            if(useGPU_)
            {
                assert(cudaSuccess == cudaMalloc((void**) &xd_, size_ * sizeof(T)));
            }
                
        }

        void copyToDevice()
        {
            assert(cudaSuccess == cudaMemcpy(xd_, x_, size_ * sizeof(T), cudaMemcpyHostToDevice));
        }

        void copyToHost()
        {
            assert(cudaSuccess == cudaMemcpy(x_, xd_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
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
            if(useGPU_) 
            {
                cudaFree(xd_);
            }              

            free(x_);
        }
};

template<typename T>
class Solver
{
    protected:
        hostCUDAVariable<T> A_, b_, x_current_, x_next_;
        const size_t resolution_;

    public:
        Solver(const size_t size, const bool useGPU) : A_(size * size, useGPU), b_(size, useGPU), 
                                                        x_current_(size, useGPU), x_next_(size, useGPU), 
                                                        resolution_(size)
        {
        }

        virtual T*& solve()
        {
        }
};

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