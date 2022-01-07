#!/bin/sh

printf "\nCleaning old files...\n"
make clean
printf "\nDone.\n"

printf "\nCompiling program...\n"
make
printf "\nDone.\n"

./solver_run

printf "\nPlotting and saving graph.\n"
python3 plot.py
printf "\nDone.\n"

