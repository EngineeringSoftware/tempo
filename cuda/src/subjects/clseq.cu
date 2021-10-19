
#include "cl.h"
#include "../mains.h"
#include "../consts.h"

/* @rivate */
__host__ void clSeqCPU(MethodSeq *ms) {
    Object objectHeap[POOL_SIZE];
    linked_list_node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    node_caching_linked_list cl(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            Object* object = env.objectAlloc();
            object->id = ms->vals[i];
            cl.add(object);
        } else {
            cl.remove(Object(ms->vals[i]));
        }
    }
}

/* @private */
__global__ void clSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    Object objectHeap[POOL_SIZE];
    linked_list_node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    node_caching_linked_list cl(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            Object* const object = env.objectAlloc();
            object->id = value;
            cl.add(object);
        } else {
            cl.remove(Object(value));
        }
    }

    _countIf(1);

    #ifdef CLSEQ_DEBUG
    if (tid == print_id) {
        // cl.print();
    }
    #endif

    #ifdef RUN_TEST
    Object* object = env.objectAlloc();
    object->id = tid;
    cl.add(object);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) clSeqGPU, clSeqCPU);
}
