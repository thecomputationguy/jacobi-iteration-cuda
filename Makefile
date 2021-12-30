CC =  gcc
#CFLAGS = -O3

all: jacobi_solver

jacobi_solver : jacobi_solver.o jacobi_cpu.o utils.o
	$(CC) $(CFLAGS) -o $@ $+ -lm

%.o : %.c utils.h jacobi_cpu.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o jacobi_solver *~

remake : clean all