#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "avlt.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void avltSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    Object objectHeap[POOL_SIZE]; 
    Node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    int_avl_tree_map tm(&env);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        if (op == 0) {
            Object* new_object = env.objectAlloc();
            new_object->id = value;
            tm.put(value, new_object);
        } else {
            tm.remove(value);
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))avltSeqOMP);
}