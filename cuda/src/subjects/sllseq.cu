
#include "sll.h"
#include "../mains.h"

/* @private */
__host__ void sllSeqCPU(MethodSeq *ms) {
    Object objectHeap[POOL_SIZE];
    SLLNode nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    SLL sll(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        Object* new_object = env.objectAlloc();
        if (ms->ops[i] == 0) {
            // TODO: when we do want to free the objects?
            sll.insertBack(new_object);
        } else {
            // TODO: do we want to use value or someting else? in remove the
            // argument is an index
            sll.remove(ms->vals[i]);
        }
    }
}

/* @private */
__global__ void sllSeqGPU(const int32_t bck_active, const int n, const int print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    Object objectHeap[POOL_SIZE];
    SLLNode nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    SLL sll(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        Object* const new_object = env.objectAlloc();
        if (op == 0) {
            // TODO: when we do want to free the objects?
            sll.insertBack(new_object);
        } else {
            // TODO: do we want to use value or someting else? in remove the
            // argument is an index
            sll.remove(value);
        }
    }

    _countIf(1);

    #ifdef SLLSEQ_DEBUG
    // if (tid == print_id) {
    //     sll.print();
    // }
    #endif

    #ifdef RUN_TEST
    Object* new_object = env.objectAlloc();
    sll.insertBack(new_object);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) sllSeqGPU, sllSeqCPU);
}
