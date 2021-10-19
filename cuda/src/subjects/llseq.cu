
#include "ll.h"
#include "../mains.h"

/* @private */
__host__ void llSeqCPU(MethodSeq *ms) {
    Object objectHeap[POOL_SIZE];
    linked_list_node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    linked_list ll(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            Object *object = env.objectAlloc();
            object->id = ms->vals[i];
            ll.add(object);
        } else {
            ll.remove(Object(ms->vals[i]));
        }
    }
}

/* @private */
__global__ void llSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    Object objectHeap[POOL_SIZE];
    linked_list_node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    linked_list ll(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            Object *const object = env.objectAlloc();
            object->id = value;
            ll.add(object);
        } else {
            ll.remove(Object(value));
        }
    }

    _countIf(1);

    #ifdef LLSEQ_DEBUG
    if (tid == print_id) {
        // ll.print();
    }
    #endif

    #ifdef RUN_TEST
    Object *object = env.objectAlloc();
    object->id = tid;
    ll.add(object);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) llSeqGPU, llSeqCPU);
}
