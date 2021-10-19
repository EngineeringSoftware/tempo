#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "rbt.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void rbtSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    Object objectHeap[POOL_SIZE];
    node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    int_red_black_tree rb(&env);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        if (op == 0) {
            Object *object = env.objectAlloc();
            object->id = value;
            rb.put(value, object);
        } else {
            rb.remove(value);
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))rbtSeqOMP);
}
