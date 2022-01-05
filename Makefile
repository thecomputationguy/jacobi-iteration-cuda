NVCC =  nvcc
CFLAGS = -O3 -gencode arch=compute_61,code=sm_61
all : jacobi_solver

main : jacobi_solver.o utils.o jacobi_CPU.o jacobi_GPU.o 

jacobi_CPU.o : jacobi_CPU.hpp jacobi_CPU.cpp
	$(NVCC) $(CFLAGS) -std=c++11 -c jacobi_CPU.cpp

jacobi_GPU.o : jacobi_GPU.cuh jacobi_GPU.cu
	$(NVCC) $(CFLAGS) -std=c++11 -c jacobi_GPU.cu

utils.o : utils.hpp utils.cpp
	$(NVCC) $(CFLAGS) -std=c++11 -c utils.cpp

jacobi_solver : jacobi_solver.cu utils.o jacobi_CPU.o jacobi_GPU.o
	$(NVCC) $(CFLAGS) -std=c++11 -o jacobi_solver jacobi_solver.cu utils.o jacobi_CPU.o jacobi_GPU.o

clean :
	rm -f *.o *~ jacobi_solver

remake : clean all