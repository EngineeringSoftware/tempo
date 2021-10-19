
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../errors.h"
#include "../explore.h"

void farrPrint(float *array, int len) {
    for (int i = 0; i < len; i++) {
        printf("%g ", array[i]);
    }
    printf("\n");
}

// https://people.cs.pitt.edu/~melhem/courses/xx45p/cuda_examples.pdf
__global__ void saxpyParallel(int n, float alpha, float *x, float *y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) y[i] = alpha * x[i] + y[i];
}

typedef struct {
    int n;
    float alpha;
    float *x;
    float *y;
} SaxpyArgs;

__global__ void saxpyGpuchoice(void *_args) {
    SaxpyArgs *args = (SaxpyArgs*) _args;

    int i = _choice(0, args->n - 1);
    args->y[i] = args->alpha * args->x[i] + args->y[i];
}

int main(void) {
    float *x = NULL;
    float *y = NULL;
    int len = 10;
    CUDA_MALLOC_MANAGED(&x, len * sizeof(float));
    CUDA_MALLOC_MANAGED(&y, len * sizeof(float));
    for (int i = 0; i < len; i++) {
        x[i] = rand() % 10;
        y[i] = rand() % 20;
    }
    farrPrint(x, len);
    farrPrint(y, len);

    saxpyParallel<<<1,100>>>(len, 2, x, y);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    farrPrint(y, len);

    SaxpyArgs *saxpy_args = NULL;
    CUDA_MALLOC_MANAGED(&saxpy_args, sizeof(SaxpyArgs));
    saxpy_args->n = len;
    saxpy_args->x = x;
    saxpy_args->y = y;
    saxpy_args->alpha = 2;
    BACKTRACK((void (*)(...))saxpyGpuchoice, saxpy_args, 1);
    farrPrint(y, len);

    CUDA_FREE(x);
    CUDA_FREE(y);
    CUDA_FREE(saxpy_args);

    return 0;
}
