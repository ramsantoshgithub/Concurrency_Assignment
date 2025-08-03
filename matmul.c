#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define N 1024
#define NUM_THREADS 8
#define BLOCK_SIZE 16
#define TILE_SIZE 32
#define ITERATIONS 1

typedef struct {
    const double *A;
    const double *B;
    double *C;
    size_t start_row;
    size_t end_row;
} ThreadData;

void initialize_matrix(double *M, size_t size) {
    for (size_t i = 0; i < size * size; i++) {
        M[i] = (double)rand() / RAND_MAX;
    }
}

void zero_matrix(double *M, size_t size) {
    for (size_t i = 0; i < size * size; i++) {
        M[i] = 0.0;
    }
}

void serial_gemm(const double *A, const double *B, double *C, size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

void *parallel_gemm(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (size_t i = data->start_row; i < data->end_row; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < N; k++) {
                sum += data->A[i * N + k] * data->B[k * N + j];
            }
            data->C[i * N + j] = sum;
        }
    }
    pthread_exit(NULL);
}

void parallel_gemm_wrapper(const double *A, const double *B, double *C) {
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    size_t rows_per_thread = N / NUM_THREADS;
    for (size_t i = 0; i < NUM_THREADS; i++) {
        thread_data[i] = (ThreadData){A, B, C, i * rows_per_thread, (i + 1) * rows_per_thread};
        if (i == NUM_THREADS - 1) {
            thread_data[i].end_row = N;
        }
        pthread_create(&threads[i], NULL, parallel_gemm, &thread_data[i]);
    }
    for (size_t i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

void blocked_gemm(const double *A, const double *B, double *C, size_t size) {
    for (size_t i = 0; i < size; i += BLOCK_SIZE) {
        for (size_t j = 0; j < size; j += BLOCK_SIZE) {
            for (size_t k = 0; k < size; k += BLOCK_SIZE) {
                for (size_t ii = i; ii < i + BLOCK_SIZE && ii < size; ii++) {
                    for (size_t jj = j; jj < j + BLOCK_SIZE && jj < size; jj++) {
                        double sum = 0.0;
                        for (size_t kk = k; kk < k + BLOCK_SIZE && kk < size; kk++) {
                            sum += A[ii * size + kk] * B[kk * size + jj];
                        }
                        C[ii * size + jj] += sum;
                    }
                }
            }
        }
    }
}

void tiled_gemm(const double *A, const double *B, double *C, size_t size) {
    for (size_t i = 0; i < size; i += TILE_SIZE) {
        for (size_t j = 0; j < size; j += TILE_SIZE) {
            for (size_t k = 0; k < size; k += TILE_SIZE) {
                for (size_t ii = i; ii < i + TILE_SIZE && ii < size; ii++) {
                    for (size_t jj = j; jj < j + TILE_SIZE && jj < size; jj++) {
                        double sum = 0.0;
                        for (size_t kk = k; kk < k + TILE_SIZE && kk < size; kk++) {
                            sum += A[ii * size + kk] * B[kk * size + jj];
                        }
                        C[ii * size + jj] += sum;
                    }
                }
            }
        }
    }
}

void benchmark(const char *name, void (*gemm_func)(const double*, const double*, double*), const double *A, const double *B, double *C, size_t matrix_size, int iterations) {
    clock_t start_time, end_time;
    double elapsed_time = 0.0, cpu_time = 0.0;

    for (int i = 0; i < iterations; i++) {
        zero_matrix(C, matrix_size);
        start_time = clock();
        gemm_func(A, B, C);
        end_time = clock();
        elapsed_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;
        cpu_time += (double)(end_time - start_time) / CLOCKS_PER_SEC;
    }

    printf("%-25s %9.3f ms   %9.3f ms       %d\n", 
           name, (elapsed_time / iterations) * 1000, 
           (cpu_time / iterations) * 1000, iterations);
}

int main() {
    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    initialize_matrix(A, N);
    initialize_matrix(B, N);

    printf("Benchmark                   Time (ms)       CPU (ms)       Iterations\n");

    benchmark("Serial GEMM", serial_gemm, A, B, C, N, ITERATIONS);
    benchmark("Parallel GEMM", parallel_gemm_wrapper, A, B, C, N, ITERATIONS);
    benchmark("Blocked GEMM", blocked_gemm, A, B, C, N, ITERATIONS);
    benchmark("Tiled GEMM", tiled_gemm, A, B, C, N, ITERATIONS);

    free(A);
    free(B);
    free(C);

    return 0;
}
