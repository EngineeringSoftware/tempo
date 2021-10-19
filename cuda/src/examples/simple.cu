
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../errors.h"
#include "../explore.h"

__device__ int demoChoice(int min, int max) {
    return _choice(min, max);
}

__global__ void simple(void *args) {
    int sum = 0;
    int i = demoChoice(0, 2);
    sum += i;
    if (i == 2) {
        int j = demoChoice(3, 3);
        sum += j;
    }
    printf("val = %d\n", sum);
}

typedef struct {
    int sum;
} SimpleReturnArgs;

__global__ void simpleReturn(void *args) {
    SimpleReturnArgs *simple_return_args = (SimpleReturnArgs*)args;
    int sum = 0;
    int i = demoChoice(0, 2);
    sum += i;
    if (i == 2) {
        int j = demoChoice(3, 3);
        sum += j;
    }
    atomicAdd(&simple_return_args->sum, sum);
}

int main(void) {
    // example with input/output arg
    // input is sum inited to 0
    // output is sum of sums
    SimpleReturnArgs *simple_return_args = NULL;
    CUDA_MALLOC_MANAGED(&simple_return_args, sizeof(SimpleReturnArgs));
    simple_return_args->sum = 0;
    BACKTRACK((void (*)(...)) simpleReturn, simple_return_args, 1);
    printf("sum = %d\n", simple_return_args->sum);
    CUDA_FREE(simple_return_args);

    return 0;
}
