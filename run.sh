#!/bin/bash

mpicc jacobi.c -o main
mpirun -np 4 ./main
rm main
