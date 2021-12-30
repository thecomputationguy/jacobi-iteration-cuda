CC =  nvcc
ARCH=61
CFLAGS=-O3 -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
LIBS=-lm

SRC_DIR = src
OBJ_DIR = bin
INC_DIR = include

EXEC = jacobi_solver

OBJS = $(OBJ_DIR)/jacobi_solver.o $(OBJ_DIR)/jacobi_cpu.o $(OBJ_DIR)/jacobi_gpu.o  

$(EXEC) : $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@

$(OBJ_DIR)/%.o : %.cu
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f bin/* *.o $(EXE) *~

remake : clean all