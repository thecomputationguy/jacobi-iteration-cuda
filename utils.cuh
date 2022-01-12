#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
template <typename T>
class hostCUDAVariable
{
    private:
        /* data */
        T* x_ ;
        T* xd_ ;
        const size_t size_;
        const bool useGPU_;
    
    public:       
        hostCUDAVariable(const size_t size, const bool useGPU);
        void copyToDevice();
        void copyToHost();
        T*& getDeviceVariable();
        T*& getHostVariable();
        ~hostCUDAVariable();
};

template<typename T>
class Solver
{
    protected:
        hostCUDAVariable<T> A_, b_, x_current_, x_next_;
        const size_t resolution_;

    public:
        Solver(const size_t size, const bool useGPU);
        virtual T*& solve();
};

inline std::vector<int> read_file();

#endif