#include <stdint.h>

#include "avlt.h"
#include "../mains.h"
#include "../consts.h"

/* @private */
__host__ void avltSeqCPU(MethodSeq *ms) {
    Object objectHeap[POOL_SIZE];
    Node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);
    int_avl_tree_map tm(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            Object* new_object = env.objectAlloc();
            new_object->id = ms->vals[i];
            tm.put(ms->vals[i], new_object);
        } else {
            tm.remove(ms->vals[i]);
        }
    }
}

/* @private */
__global__ void avltSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    Object objectHeap[POOL_SIZE];
    Node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    int_avl_tree_map tm(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            Object* const new_object = env.objectAlloc();
            new_object->id = value;
            tm.put(value, new_object);
        } else {
            tm.remove(value);
        }
    }

    _countIf(1);

    #ifdef AVLTSEQ_DEBUG
    if (tid == print_id) {
        // tm.print();
    }
    #endif

    #ifdef RUN_TEST
    Object* new_object = env.objectAlloc();
    tm.put(tid, new_object);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) avltSeqGPU, avltSeqCPU);
}
