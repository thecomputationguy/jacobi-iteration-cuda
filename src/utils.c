#include<stdlib.h>
#include "include/utils.h"

typedef struct
{
    int Nx, Ny;
    float* A;
    float* x_current;
    float* x_next;
    float* b;
} parameters;

void initializeData(parameters* P, const int Nx, const int Ny)
{
    P->Nx = Nx;
    P->Ny = Ny;
    P->x_current = (float*)malloc(Nx * sizeof(float));
    P->x_next = (float*)malloc(Nx * sizeof(float));
    P->b = (float*)malloc(Nx * sizeof(float));
    P->A = (float*)malloc(Nx * Ny * sizeof(float));
}