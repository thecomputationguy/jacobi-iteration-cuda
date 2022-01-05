NVCC =  nvcc
CFLAGS = -O3 -gencode arch=compute_61,code=sm_61
all : solver_run

main : solver_run.o solvers.o utils.o jacobi_CPU.o jacobi_GPU.o 

jacobi_CPU.o : jacobi_CPU.hpp jacobi_CPU.cpp
	$(NVCC) $(CFLAGS) -std=c++11 -c jacobi_CPU.cpp

jacobi_GPU.o : jacobi_GPU.cuh jacobi_GPU.cu
	$(NVCC) $(CFLAGS) -std=c++11 -c jacobi_GPU.cu

utils.o : utils.cuh utils.cu
	$(NVCC) $(CFLAGS) -std=c++11 -c utils.cu

solvers.o : solvers.cuh solvers.cu
	$(NVCC) $(CFLAGS) -std=c++11 -c solvers.cu

solver_run : solver_run.cu solvers.o utils.o jacobi_CPU.o jacobi_GPU.o
	$(NVCC) $(CFLAGS) -std=c++11 -o solver_run solver_run.cu solvers.o utils.o jacobi_CPU.o jacobi_GPU.o

clean :
	rm -f *.o *~ solver_run

remake : clean all