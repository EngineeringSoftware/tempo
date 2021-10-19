
#include "errors.h"
#include <stdio.h>

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void checkCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess) {
        return;
    }
    fprintf(stderr, "%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err), err, file, line);
    exit(1);
}
