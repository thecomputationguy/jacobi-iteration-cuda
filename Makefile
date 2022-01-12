NVCC =  nvcc 
CFLAGS = -O3 -Xcompiler=-fopenmp -gencode arch=compute_53,code=sm_53
all : solver_run

jacobi_CPU.o : jacobi_CPU.hpp jacobi_CPU.cpp
	$(NVCC) $(CFLAGS) -c jacobi_CPU.cpp

jacobi_GPU.o : jacobi_GPU.cuh jacobi_GPU.cu
	$(NVCC) $(CFLAGS) -c jacobi_GPU.cu

utils.o : utils.cuh utils.cu
	$(NVCC) $(CFLAGS) -c utils.cu

solvers.o : solvers.cuh solvers.cu
	$(NVCC) $(CFLAGS) -c solvers.cu

solver_run : main.cu solvers.o utils.o jacobi_CPU.o jacobi_GPU.o
	$(NVCC) $(CFLAGS) -o solver_run main.cu solvers.o utils.o jacobi_CPU.o jacobi_GPU.o

clean :
	rm -f *.o *~ solver_run

remake : clean all