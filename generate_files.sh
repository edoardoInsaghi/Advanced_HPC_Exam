#!/bin/bash

for PROCS in 1 2; do
  cat > jacobi_mpi_${PROCS}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=jacobi_${PROCS}mpi
#SBATCH --ntasks=${PROCS}
#SBATCH --nodes=${PROCS}
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --partition=EPYC

module load openMPI
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

for NX in 1000 10000; do
  for THREADS in 1 2 4 8 16 32 64 128; do
    echo "Running NX=\$NX THREADS=\$THREADS with ${PROCS} MPI processes"
    export OMP_NUM_THREADS=\$THREADS
    mpirun -np ${PROCS} ./test_cpu \$NX \$THREADS
  done
done
EOF
done
