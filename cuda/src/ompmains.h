#ifndef OMP_MAINS_H
#define OMP_MAINS_H

#include <stdio.h>
#include "omp.h"
#include "explore.h"

int seqMainOMP(int argc, char *argv[], void (*kernel)(...)) {
    if (argc < 2) {
        printf("Incorrect arguments: size\n");
        exit(1);
    }

    int32_t size = atoi(argv[1]);

    int* args = (int*) malloc(sizeof(int));
    args[0] = size;

    double start = omp_get_wtime();
    explore(kernel, (void*)args, 1);
    double stop = omp_get_wtime();
    float time_spent = (stop - start) * 1000;

    printf("Driver time: %.2lf\n", time_spent);

    free(args);

    return 0;
}

int uditaMainOMP(int argc, char *argv[], void (*kernel)(...)) {
    if (argc < 2) {
        printf("Incorrect arguments: size\n");
        exit(1);
    }

    int32_t size = atoi(argv[1]);

    int* args = (int*) malloc(sizeof(int));
    args[0] = size;

    double start = omp_get_wtime();
    explore(kernel, (void*)args, 1);
    double stop = omp_get_wtime();
    float time_spent = (stop - start) * 1000;

    printf("Driver time: %.2lf\n", time_spent);

    free(args);

    return 0;   
}

#endif