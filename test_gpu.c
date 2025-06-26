#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define STEPS 2000
#define IDX(i,j) ((i)*(NX+2)+(j))

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int NX = atoi(argv[1]);
    int teams = atoi(argv[2]);            // GPU thread teams
    int threads_per_team = atoi(argv[3]); // Threads per team
    int NY = NX;

    double setup_start = MPI_Wtime();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Domain decomposition
    int base_rows = NY / size;
    int extra = NY % size;
    int local_rows = base_rows + (rank < extra ? 1 : 0);
    int start_row = rank * base_rows + (rank < extra ? rank : extra);
    int total_rows = local_rows + 2;
    int row_size = NX + 2;

    // Allocate host memory
    double *u = calloc(total_rows * row_size, sizeof(double));
    double *u_new = calloc(total_rows * row_size, sizeof(double));
    double temp = 100.0;
    memset(u, 0, total_rows * row_size * sizeof(double));
    memset(u_new, 0, total_rows * row_size * sizeof(double));
    
    // Initialize boundaries
    if (rank == 0) {
        for (int j = 0; j < row_size; j++) {
            u[IDX(0, j)] = temp;
            u_new[IDX(0, j)] = temp;
        }
    }
    for (int i = 0; i < total_rows; i++) {
        u[IDX(i, 0)] = temp;
        u_new[IDX(i, 0)] = temp;
    }

    // Allocate device memory
    #pragma omp target enter data map(to: u[0:total_rows*row_size], u_new[0:total_rows*row_size])

    // Create MPI window
    MPI_Win win;
    MPI_Win_create(u, total_rows * row_size * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    
    double setup_end = MPI_Wtime();
    double setup_time = setup_end - setup_start;
    double comm_time = 0.0, comp_time = 0.0;

    for (int step=0; step<STEPS; step++) {
        
        comm_start = MPI_Wtime();
        //#pragma omp target update from(u_old[IDX(1,0):row_size])          // First interior row
        //#pragma omp target update from(u_old[IDX(local_rows,0):row_size]) // Last interior row
        MPI_Win_fence(0, win);
        MPI_Get(&u_old[IDX(0,0)], row_size, MPI_DOUBLE, up, IDX(local_rows,0), row_size, MPI_DOUBLE, win);
        MPI_Get(&u_old[IDX(local_rows+1,0)], row_size, MPI_DOUBLE, down, IDX(1,0), row_size, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);
        //#pragma omp target update to(u_old[IDX(0,0):row_size])            // Top ghost
        //#pragma omp target update to(u_old[IDX(local_rows+1,0):row_size]) // Bottom ghost
        comm_end = MPI_Wtime();
        comm_time += comm_end - comm_start;

        // GPU Kernel
        comp_start = MPI_Wtime();
        #pragma omp target teams num_teams(teams) thread_limit(threads_per_team) is_device_ptr(u_old, u_next)
        #pragma omp distribute parallel for collapse(2)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= NX; j++) {
                u_next[IDX(i,j)] = 0.25 * (
                    u_old[IDX(i+1,j)] + u_old[IDX(i-1,j)] +
                    u_old[IDX(i,j+1)] + u_old[IDX(i,j-1)]
                );
            }
        }
        comp_end = MPI_Wtime();
        comp_time += comp_end - comp_start;

        double* tmp = u;
        u = u_next;
        u_next = tmp;
    }

    // Output timings
    char filename[64];
    sprintf(filename, "timing_size_%d_rank_%d_NX_%d_teams_%d_threads_%d_gpu.csv", 
            size, rank, NX, teams, threads_per_team);
    FILE* fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "%d,%d,%d,%d,%d,%f,%f,%f\n", 
                size, rank, NX, teams, threads_per_team, setup_time, comm_time, comp_time);
        fclose(fp);
    }

    // Cleanup
    #pragma omp target exit data map(delete: u[0:total_rows*row_size], u_new[0:total_rows*row_size])
    free(u); free(u_new);
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}