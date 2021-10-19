#include <stdio.h>
#include <stdlib.h>
#include "../errors.h"
#include "../explore.h"

#define MATRIX_DIMENSION 10
#define IDX(x, y) ((x) * MATRIX_DIMENSION + (y))

void matrixPrint(int* matrix) {
    for (int i = 0; i < MATRIX_DIMENSION; i++) {
        for (int j = 0; j < MATRIX_DIMENSION; j++) {
            printf("%d ", matrix[IDX(i, j)]);
        }
        printf("\n");
    }
}

__global__ void matrixMultiply(int* matrixA, int* matrixB, int* result) {
    int x = _choice(0, MATRIX_DIMENSION - 1);
    int y = _choice(0, MATRIX_DIMENSION - 1);
    int z = _choice(0, MATRIX_DIMENSION - 1);
    
    int product = matrixA[IDX(x, z)] * matrixB[IDX(z, y)];
    atomicAdd(&result[IDX(x, y)], product);
}

int main() {
    int* matrixA;
    int* matrixB;
    int* result;
    CUDA_MALLOC_MANAGED(&matrixA, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int));
    CUDA_MALLOC_MANAGED(&matrixB, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int));
    CUDA_MALLOC_MANAGED(&result, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(int));
    
    for (int i = 0; i < MATRIX_DIMENSION * MATRIX_DIMENSION; i++) {
        matrixA[i] = rand() % 10;
        matrixB[i] = rand() % 20;
    }

    printf("Matrix A:\n");
    matrixPrint(matrixA);
    printf("Matrix B:\n");
    matrixPrint(matrixB);

    EXPLORE(matrixMultiply<<<starting_blocks, starting_threads>>>(matrixA, matrixB, result));

    printf("Result:\n");
    matrixPrint(result);
}