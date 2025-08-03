#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    double **A;
    double **B;
} MatrixPair;

MatrixPair* buffer;
int BUFFER_SIZE, in = 0, out = 0, count = 0;
int NUM_PRODUCERS, NUM_CONSUMERS, M, K, N, NUM_OPERATIONS;
int produced_count = 0, consumed_count = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t not_full = PTHREAD_COND_INITIALIZER;
pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;

void generate_random_matrix(double **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i][j] = rand() % 10;
}

MatrixPair create_matrix_pair() {
    MatrixPair matrices;
    matrices.A = (double**)malloc(M * sizeof(double*));
    matrices.B = (double**)malloc(K * sizeof(double*));
    if (!matrices.A || !matrices.B) {
        printf("Memory allocation failed for matrix pair.\n");
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        matrices.A[i] = (double*)malloc(K * sizeof(double));
        if (!matrices.A[i]) {
            printf("Memory allocation failed for matrix A.\n");
            exit(1);
        }
    }
    for (int i = 0; i < K; i++) {
        matrices.B[i] = (double*)malloc(N * sizeof(double));
        if (!matrices.B[i]) {
            printf("Memory allocation failed for matrix B.\n");
            exit(1);
        }
    }

    generate_random_matrix(matrices.A, M, K);
    generate_random_matrix(matrices.B, K, N);
    return matrices;
}

void* producer(void* arg) {
    while (1) {
        pthread_mutex_lock(&mutex);
        
        if (produced_count >= NUM_OPERATIONS) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        
        while (count == BUFFER_SIZE)
            pthread_cond_wait(&not_full, &mutex);

        buffer[in] = create_matrix_pair();
        in = (in + 1) % BUFFER_SIZE;
        count++;
        produced_count++;
        printf("Produced matrix at buffer position %d (produced count: %d)\n", in, produced_count);

        pthread_cond_signal(&not_empty);
        pthread_mutex_unlock(&mutex);
        sleep(1);
    }
    return NULL;
}

void* consumer(void* arg) {
    double **C = (double**)malloc(M * sizeof(double*));
    for (int i = 0; i < M; i++) {
        C[i] = (double*)malloc(N * sizeof(double));
        if (!C[i]) {
            printf("Memory allocation failed for matrix C.\n");
            exit(1);
        }
    }

    while (1) {
        pthread_mutex_lock(&mutex);
        
        if (consumed_count >= NUM_OPERATIONS) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        
        while (count == 0)
            pthread_cond_wait(&not_empty, &mutex);

        MatrixPair matrices = buffer[out];
        out = (out + 1) % BUFFER_SIZE;
        count--;
        consumed_count++;
        printf("Consumed matrix from buffer position %d (consumed count: %d)\n", out, consumed_count);

        pthread_cond_signal(&not_full);
        pthread_mutex_unlock(&mutex);

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = 0;
                for (int k = 0; k < K; k++) {
                    C[i][j] += matrices.A[i][k] * matrices.B[k][j];
                }
            }
        }
        
        for (int i = 0; i < M; i++) free(matrices.A[i]);
        for (int i = 0; i < K; i++) free(matrices.B[i]);
        free(matrices.A);
        free(matrices.B);
    }

    for (int i = 0; i < M; i++) free(C[i]);
    free(C);

    return NULL;
}

int main() {
    printf("Enter buffer size: ");
    scanf("%d", &BUFFER_SIZE);
    if (BUFFER_SIZE <= 0) {
        printf("Invalid buffer size.\n");
        return 1;
    }
    
    printf("Enter number of producers and consumers: ");
    scanf("%d %d", &NUM_PRODUCERS, &NUM_CONSUMERS);
    if (NUM_PRODUCERS <= 0 || NUM_CONSUMERS <= 0) {
        printf("Invalid number of producers or consumers.\n");
        return 1;
    }
    
    printf("Enter matrix dimensions M, K, N: ");
    scanf("%d %d %d", &M, &K, &N);
    if (M <= 0 || K <= 0 || N <= 0) {
        printf("Invalid matrix dimensions.\n");
        return 1;
    }
    
    printf("Enter number of operations: ");
    scanf("%d", &NUM_OPERATIONS);
    if (NUM_OPERATIONS <= 0) {
        printf("Invalid number of operations.\n");
        return 1;
    }

    buffer = (MatrixPair*)malloc(BUFFER_SIZE * sizeof(MatrixPair));
    if (!buffer) {
        printf("Memory allocation failed for buffer.\n");
        return 1;
    }

    pthread_t prod_threads[NUM_PRODUCERS], cons_threads[NUM_CONSUMERS];

    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_create(&prod_threads[i], NULL, producer, NULL);
    }

    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_create(&cons_threads[i], NULL, consumer, NULL);
    }

    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_join(prod_threads[i], NULL);
    }

    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_join(cons_threads[i], NULL);
    }

    free(buffer);
    return 0;
}
