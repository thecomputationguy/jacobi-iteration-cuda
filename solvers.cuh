#ifndef SOLVERS_CUH
#define SOLVERS_CUH

#include "utils.cuh"
#include "utils.cu"

template<typename T>
class jacobiSolverGPU : public Solver<T>
{
    private:
        const int numBlocks_, blockSize_;
    public:
        jacobiSolverGPU(size_t resolution, const bool useGPU, const int numBlocks, const int blockSize);
        T*& solve();
};

template<typename T>
class jacobiSolverCPU : public Solver<T>
{
    public:
        jacobiSolverCPU(size_t resolution, const bool useGPU);
        T*& solve();
};


#endif