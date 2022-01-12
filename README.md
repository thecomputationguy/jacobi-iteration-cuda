# jacobi-iteration-cuda

GPU version of the Jacobi Iteration implemented using both CUDA C++ and PyCUDA

For the CUDA C++ solvers, simply run the 'run_solvers.sh' script and everything is done automatically. However, one needs to assign appropriate permissions to the shell script first. To do that, in a linux terminal, run 'chmod 777 run_solver.sh'. Then run './run_solvers.sh'. Once the runs are finished, a comparison graph is generated for the runtimes and is saved as 'plot_jacobi.png' and the runtimes are stored as a csv file in 'measurements.csv'.

The problem sizes are defined in the file 'resolutions.txt'. To add another resolution, just append it below the last one.

To Run the PyCuda version, run the notebook cells in 'jacobiPyCuda.ipynb' sequentially.
