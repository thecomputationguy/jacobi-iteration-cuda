#ifndef UTILS
#define UTILS

typedef struct
{
    int Nx, Ny;
    float* A;
    float* x_current;
    float* x_next;
    float* b;
} parameters;

template <typename T>
class hostCUDAVariable
{
private:
    /* data */
    
    T* xd_ ;
public:
    T* x_ ;
    hostCUDAVariable(const int size)
    {
        x_ = new
    };
    ~hostCUDAVariable();
};

hostCUDAVariable::hostCUDAVariable(/* args */)
{
}

hostCUDAVariable::~hostCUDAVariable()
{
}
 


#endif