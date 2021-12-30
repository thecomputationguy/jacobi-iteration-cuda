
void jacobiCPU(float* x_new, const float* A, float* x_current, float* b, const int Nx, const int Ny)
{
    int i, j;
    float sum;

    for(i = 0; i < Nx; i++)
    {
        sum = 0;
        for(j = 0; j < Ny; j++)
        {
            if(i != j)
                sum += A[i * Ny + j] * x_current[j];
        }
        x_new[i] = (b[i] - sum) / A[i * Ny + i];
    }
}