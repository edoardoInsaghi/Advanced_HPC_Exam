#!/bin/bash

for PROCS in 1 2 4 8 16 32; do
  cat > jacobi_mpi_${PROCS}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=jacobi_${PROCS}_mpi
#SBATCH --ntasks=${PROCS}
#SBATCH --nodes=${PROCS}
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ict25_mhpc_0

module load boost/1.83.0--openmpi--4.1.6--gcc--12.2.0
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

for NX in 1000 10000; do
  for THREADS in 1 2 4 8 16 32; do
    echo "Running NX=\$NX THREADS=\$THREADS with ${PROCS} MPI processes"
    export OMP_NUM_THREADS=\$THREADS
    mpirun -np ${PROCS} --map-by node --bind-to none ./test_cpu \$NX \$THREADS
  done
done
EOF
