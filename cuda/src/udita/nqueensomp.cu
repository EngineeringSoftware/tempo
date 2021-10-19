
#include <math.h>
#include <stdio.h>
#include "nqueens.h"
#include "../ompmains.h"

int8_t checkQueens(int8_t *array, int first, int second) {
    return !(array[first] == array[second] ||
             abs(first - second) == abs(array[first] - array[second]));
}

void nqueensGenerate(Env *env, NQueens *nqueens) {
    // note that in the origina .java code _choice is between (0,
    // maxSize); however, the numbers in that case do not match, so it
    // must be that we used specific size (and not up-to size).
    nqueens->size = _choice(env->max_size, env->max_size);
    // array = new int[size];

    for (int32_t i = 0; i < nqueens->size; i++) {
        nqueens->array[i] = _choice(0, nqueens->size - 1);
        for (int32_t j = 0; j < i; j++) {
            _ignoreIf(!checkQueens(nqueens->array, i, j));
        }
    }
}

int8_t nqueensRepOk(Env *env, NQueens *nqueens) {
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

void nqueensUdita(int8_t size) {
    int tid = omp_get_thread_num();

    Env env = {
        .max_size = size,
    };
    NQueens nqueens = {
        .size = 0,
    };
    nqueensGenerate(&env, &nqueens);
    _countIf(1);
}

int main(int argc, char *argv[]) {
    return uditaMainOMP(argc, argv, (void (*)(...))nqueensUdita);
}
