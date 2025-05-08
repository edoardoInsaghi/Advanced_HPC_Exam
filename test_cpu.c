#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


#define STEPS 2000

#define IDX(i,j) ((i)*(NX+2)+(j))  // 2D indexing macro with ghost cells

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int NX, NY;
    NX = atoi(argv[1]);
    NY = NX;

    int user_threads = atoi(argv[2]);
    omp_set_num_threads(user_threads);

    double setup_start, setup_end, comm_start, comm_end, comp_start, comp_end;
    double setup_time, comm_time, comp_time;
    setup_start = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int omp_threads = 0;
    #pragma omp parallel
    {
        #pragma omp master
        {
            omp_threads = omp_get_num_threads();
        }
    }
    if (rank == 0) {
        printf("MPI started with %d processes\n", size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d using %d OpenMP threads\n", rank, omp_threads);

    int base_rows = NY / size;
    int extra = NY % size;
    int local_rows = base_rows + (rank < extra ? 1 : 0);
    int start_row = rank * base_rows + (rank < extra ? rank : extra);

    int total_rows = local_rows + 2;
    int row_size = NX + 2;

    double* u = calloc(total_rows * row_size, sizeof(double));
    double* u_new = calloc(total_rows * row_size, sizeof(double));

    double temp = 100.0;
    memset(u, 0, total_rows * row_size * sizeof(double));
    memset(u_new, 0, total_rows * row_size * sizeof(double));
    if (rank == 0) {
        for (int j = 0; j < row_size; j++) {
            u[IDX(0, j)] = temp;
            u_new[IDX(0, j)] = temp;
        }
    }
    for (int i=0; i<total_rows; i++) {
        u[IDX(i, 0)] = temp;
        u_new[IDX(i, 0)] = temp;
    }

    MPI_Win win;
    MPI_Win_create(u, total_rows * row_size * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    setup_end = MPI_Wtime();
    setup_time = setup_end - setup_start;

    int quit = 0;
    for (int step=0; step<STEPS && !quit; ++step) {


        comm_start = MPI_Wtime();
        MPI_Win_fence(0, win);
        MPI_Get(&u[IDX(0,0)], NX+2, MPI_DOUBLE, up, IDX(local_rows-1,0), NX+2, MPI_DOUBLE, win);
        MPI_Get(&u[IDX(local_rows+1,0)], NX+2, MPI_DOUBLE, down, IDX(1,0), NX+2, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);
        comm_end = MPI_Wtime();
        comm_time += comm_end - comm_start;

        comp_start = MPI_Wtime();
        #pragma omp parallel for collapse(2)
        for (int i=1; i<=local_rows; ++i) {
            for (int j=1; j<=NX; ++j) {
                u_new[IDX(i,j)] = 0.25 * (
                    u[IDX(i+1,j)] + u[IDX(i-1,j)] +
                    u[IDX(i,j+1)] + u[IDX(i,j-1)]);
            }
        }
        comp_end = MPI_Wtime();
        comp_time += comp_end - comp_start;

        double* tmp = u;
        u = u_new;
        u_new = tmp;
    }

    char filename[64];
    sprintf(filename, "timing_size_%d_rank_%d_NX_%d_threads_%d_cpu.csv", size, rank, NX, omp_threads);
    FILE* fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "%d,%d,%d,%d,%f,%f,%f\n", size, rank, NX, omp_threads, setup_time, comm_time, comp_time);
        fclose(fp);
    } else {
        fprintf(stderr, "Rank %d: Failed to write timing file\n", rank);
    }

    free(u);
    free(u_new);
    MPI_Win_free(&win);
    MPI_Finalize();

    return 0;
}
