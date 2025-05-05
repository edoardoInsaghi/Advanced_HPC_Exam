#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL.h>

#define NX 200
#define NY 200
#define STEPS 20000
#define CELL_SIZE 3  // Size of each cell in pixels

#define IDX(i,j) ((i)*(NX+2)+(j))  // 2D indexing macro with ghost cells

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

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

    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;
    if (rank == 0) {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        window = SDL_CreateWindow("Heat Diffusion", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  NX * CELL_SIZE, NY * CELL_SIZE, SDL_WINDOW_SHOWN);
        if (!window) {
            fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
            SDL_Quit();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer) {
            fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError());
            SDL_DestroyWindow(window);
            SDL_Quit();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int base_rows = NY / size;
    int extra = NY % size;
    int local_rows = base_rows + (rank < extra ? 1 : 0);
    int start_row = rank * base_rows + (rank < extra ? rank : extra);

    int total_rows = local_rows + 2;
    int row_size = NX + 2;

    double* u = calloc(total_rows * row_size, sizeof(double));
    double* u_new = calloc(total_rows * row_size, sizeof(double));

    double temp = 10000.0;
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

    double* full_matrix = NULL;
    int* recvcounts = NULL;
    int* displs = NULL;
    int* all_rows = malloc(size * sizeof(int));
    for (int r=0; r<size; ++r) {
        all_rows[r] = base_rows + (r < extra ? 1 : 0);
    }

    if (rank == 0) {
        full_matrix = malloc(NY * (NX + 2) * sizeof(double));
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        displs[0] = 0;
        for (int r = 0; r < size; ++r) {
            recvcounts[r] = all_rows[r] * (NX + 2);
            if (r > 0) displs[r] = displs[r-1] + recvcounts[r-1];
        }
    }

    int quit = 0;
    for (int step=0; step<STEPS && !quit; ++step) {

        MPI_Win_fence(0, win);
        MPI_Get(&u[IDX(0,1)], NX, MPI_DOUBLE, up, IDX(local_rows-1,1), NX, MPI_DOUBLE, win);
        MPI_Get(&u[IDX(local_rows+1,1)], NX, MPI_DOUBLE, down, IDX(1,1), NX, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);

        #pragma omp parallel for collapse(2)
        for (int i=1; i<=local_rows; ++i) {
            for (int j=1; j<=NX; ++j) {
                u_new[IDX(i,j)] = 0.25 * (
                    u[IDX(i+1,j)] + u[IDX(i-1,j)] +
                    u[IDX(i,j+1)] + u[IDX(i,j-1)]);
            }
        }

        double* tmp = u;
        u = u_new;
        u_new = tmp;

        MPI_Gatherv(&u[IDX(1,0)], local_rows * (NX + 2), MPI_DOUBLE,
                    full_matrix, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    quit = 1;
                }
            }

            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);

            for (int i=0; i<NY; ++i) {
                for (int j=1; j<=NX; ++j) {
                    double value = full_matrix[IDX(i,j)];
                    uint8_t red = (uint8_t)((value / temp) * 255);
                    uint8_t blue = (uint8_t)((1.0 - value / temp) * 255);
                    SDL_SetRenderDrawColor(renderer, red, 0, blue, 255);
                    SDL_Rect rect = {
                        (j-1) * CELL_SIZE,
                        i * CELL_SIZE,
                        CELL_SIZE, 
                        CELL_SIZE
                    };
                    SDL_RenderFillRect(renderer, &rect);
                }
            }

            SDL_RenderPresent(renderer);
        } 

        MPI_Bcast(&quit, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (step % 1000 == 0) {
                printf("Step %d\n", step);
            }
        }
    }

    free(u);
    free(u_new);
    free(all_rows);
    if (rank == 0) {
        free(full_matrix);
        free(recvcounts);
        free(displs);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
    MPI_Win_free(&win);
    MPI_Finalize();

    return 0;
}
