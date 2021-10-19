
#include <stdio.h>
#include "ha.h"
#include "../mains.h"

// Needed for running a test

__device__ int haRemove(Env *env, HeapArray *ha) {
    if (ha->size == 0) {
        return -1;
    }

    int ret = ha->array[0];
    ha->array[0] = ha->array[ha->size - 1];

    ha->size--;
    int i = 0;
    while (ha->size != 0) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left >= ha->size) {
            break;
        }

        if (ha->array[i] < ha->array[left]) {
            int temp = ha->array[i];
            ha->array[i] = ha->array[left];
            ha->array[left] = temp;
            i = left;
            continue;
        }

        if (right >= ha->size) {
            break;
        }

        if (ha->array[i] < ha->array[right]) {
            int temp = ha->array[i];
            ha->array[i] = ha->array[right];
            ha->array[right] = temp;
            i = right;
        }
        ha->size--;
    }

    return ret;
}

// ----------------------------------------

// LOC.py start
__device__ void haGenerate(Env *const env, HeapArray *const ha) {
    ha->arraylength = _choice(0, env->max_array_length);
    ha->size = _choice(0, ha->arraylength);

    if (ha->size != 0) {
        ha->array[0] = _choice(0, env->max_array_length);
    }

    for (int32_t i = 1; i < ha->size; i++) {
        ha->array[i] = _choice(0, ha->array[(i-1) >> 1]);
    }
}

__global__ void haUdita(const int32_t bck_active, const int32_t size) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    Env env = {.max_array_length = size,};

    HeapArray ha;
    haGenerate(&env, &ha);
    _countIf(1);

    // LOC.py stop
    #ifdef RUN_TEST
    int removed = haRemove(&env, &ha);
    #endif
}

int main(int argc, char *argv[]) {
    return uditaMain(argc, argv, (void (*)(...)) haUdita);
}
