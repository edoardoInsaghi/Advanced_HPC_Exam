#!/bin/bash

for PROCS in 1 2 4 8 16 32; do
  cat > jacobi_gpu_${PROCS}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=jacobi_gpu_${PROCS}
#SBATCH --ntasks=${PROCS}
#SBATCH --gpus-per-task=1
#SBATCH --nodes=${PROCS}
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ict25_mhpc_0

module load openMPI/4.1.5
module load cuda

for NX in 1000 10000; do
  for TEAMS in 64 128 256; do
    for THREADS_PER_TEAM in 32 64 128; do
      echo "Running NX=\${NX} with ${PROCS} GPUs, \${TEAMS} teams, \${THREADS_PER_TEAM} threads/team"
      mpirun -np ${PROCS} ./test_gpu \${NX} \${TEAMS} \${THREADS_PER_TEAM}
    done
  done
done
EOF
done