
#ifndef MAINS_H
#define MAINS_H

#include <assert.h>
#include <stdio.h>
#include "explore.h"
#include "errors.h"
#include "subjects/sequtil.h"

/* @private */
int seqMainGPU(int size, int print_id, void (*kernel)(...)) {
    // need this for some GPUs to establish context (and do more
    // proper time measurment)
    cudaFree(0);

    size_t limit;
    CUDA_CHECK_RETURN(cudaDeviceGetLimit(&limit, cudaLimitStackSize));
    CUDA_CHECK_RETURN(cudaDeviceSetLimit(cudaLimitStackSize, limit * 50));

    float time;
    cudaEvent_t start, stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    int* args = (int*) malloc(2 * sizeof(int));
    args[0] = size;
    args[1] = print_id;

    CUDA_CHECK_RETURN(cudaEventRecord(start));
    explore(kernel, (void*)args, 2);
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);

    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
    free(args);

    return 0;
}

/* @public */
int seqMain(int argc, char *argv[], void (*gpuKernel)(...), void (*cpuKernel)(MethodSeq*)) {
    if (argc < 2) {
        printf("Incorrect arguments: size [print_id]\n");
        exit(1);
    }
    int size = atoi(argv[1]);

    int print_id = -1;
    if (argc > 2) {
        print_id = atoi(argv[2]);
    }

    // we use negative to indicate CPU runs
    if (size >= 0) {
        return seqMainGPU(size, print_id, gpuKernel);
    } else {
        return seqMainCPU(-size, cpuKernel);
    }
}

/* @public */
int uditaMain(int argc, char *argv[], void (*kernel)(...)) {
    if (argc < 2) {
        printf("Incorrect arguments: size\n");
        exit(1);
    }
    int32_t size = atoi(argv[1]);

    // establish context
    cudaFree(0);

    float time;
    cudaEvent_t start, stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    int* args = (int*) malloc(sizeof(int));
    args[0] = size;

    CUDA_CHECK_RETURN(cudaEventRecord(start));
    explore(kernel, (void*)args, 1);
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);
    
    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
    free(args);

    return 0;
}

#endif
