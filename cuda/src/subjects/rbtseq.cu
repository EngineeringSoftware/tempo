
#include "rbt.h"
#include "../mains.h"

/* @private */
__host__ void rbtSeqCPU(MethodSeq *ms) {
    Object objectHeap[POOL_SIZE];
    node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    int_red_black_tree rb(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            Object *object = env.objectAlloc();
            object->id = ms->vals[i];
            rb.put(ms->vals[i], object);
        } else {
            rb.remove(ms->vals[i]);
        }
    }
}

/* @private */
__global__ void rbtSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    Object objectHeap[POOL_SIZE];
    node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    int_red_black_tree rb(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            Object *const object = env.objectAlloc();
            object->id = value;
            rb.put(value, object);
        } else {
            rb.remove(value);
        }
    }

    _countIf(1);

    #ifdef RBTREESEQ_DEBUG
    if (tid == print_id) {
        // rb.print();
    }
    #endif

    #ifdef RUN_TEST
    Object *object = env.objectAlloc();
    object->id = tid;
    rb.put(tid, object);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) rbtSeqGPU, rbtSeqCPU);
}
