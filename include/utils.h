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

void initializeData(parameters P, const int Nx, const int Ny);

#endif