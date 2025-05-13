#!/bin/bash

for PROCS in 1 2 4 8; do
  sbatch jacobi_mpi_${PROCS}.sbatch
done
