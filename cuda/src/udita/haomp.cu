
#include <stdio.h>
#include "ha.h"
#include "../ompmains.h"

void haGenerate(Env* env, HeapArray *ha) {
    ha->arraylength = _choice(0, env->max_array_length);
    ha->size = _choice(0, ha->arraylength);

    if (ha->size != 0) {
        ha->array[0] = _choice(0, env->max_array_length);
    }

    for (int32_t i = 1; i < ha->size; i++) {
        ha->array[i] = _choice(0, ha->array[(i-1) >> 1]);
    }
}

void haUdita(int32_t size) {
    int tid = omp_get_thread_num();

    Env env = {
        .max_array_length = size,
    };

    HeapArray ha;
    haGenerate(&env, &ha);
    _countIf(1);
}

int main(int argc, char *argv[]) {
    return uditaMainOMP(argc, argv, (void (*)(...))haUdita);
}
