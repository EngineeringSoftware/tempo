
#include <math.h>
#include <stdio.h>
#include "nqueens.h"
#include "../mains.h"

// LOC.py start
__device__ int8_t checkQueens(const int8_t *const array, const int8_t first, const int8_t second) {
    return !(array[first] == array[second] ||
             abs(first - second) == abs(array[first] - array[second]));
}

__device__ void nqueensGenerate(const Env *const env, NQueens *const nqueens) {
    // note that in the origina .java code _choice is between (0,
    // maxSize); however, the numbers in that case do not match, so it
    // must be that we used specific size (and not up-to size).
    nqueens->size = _choice(env->max_size, env->max_size);
    // array = new int[size];

    for (int8_t i = 0; i < nqueens->size; i++) {
        nqueens->array[i] = _choice(0, nqueens->size - 1);
        for (int8_t j = 0; j < i; j++) {
            _ignoreIf(!checkQueens(nqueens->array, i, j));
        }
    }
}

__device__ int8_t nqueensRepOk(const Env *const env, NQueens *const nqueens) {
    if (nqueens->size < 0 || nqueens->size > env->max_size) {
        return 0;
    }
    if (nqueens->array == NULL) {
        return 0;
    }
    // if (array.length != size) return false;

    for (int32_t i = 0; i < nqueens->size; i++) {
        if (nqueens->array[i] < 0 || nqueens->array[i] > nqueens->size - 1) {
            return 0;
        }
        for (int32_t j = 0; j < i; j++) {
            if (!checkQueens(nqueens->array, i, j)) {
                return 0;
            }
        }
    }
    return 1;
}

__global__ void nqueensUdita(const int32_t bck_active, const int8_t size) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    Env env = {
        .max_size = size,
    };
    NQueens nqueens = {
        .size = 0,
    };
    nqueensGenerate(&env, &nqueens);
    _countIf(1);

    #ifdef RUN_TEST
    nqueensRepOk(&env, &nqueens);
    #endif
}
// LOC.py stop


int main(int argc, char *argv[]) {
    return uditaMain(argc, argv, (void (*)(...)) nqueensUdita);
}
