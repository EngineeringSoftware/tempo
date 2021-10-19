
#include "bh.h"
#include "../mains.h"
#include "../consts.h"

/* @private */
__host__ void bhSeqCPU(MethodSeq *ms) {
    binomial_heap_node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    binomial_heap bh(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            bh.insert(ms->vals[i]);
        } else {
            bh.deleteNode(ms->vals[i]);
        }
    }
}

/* @private */
__global__ void bhSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    binomial_heap_node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    binomial_heap bh(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            bh.insert(value);
        } else {
            bh.deleteNode(value);
        }
    }

    _countIf(1);

    #ifdef BHSEQ_DEBUG
    if (tid == print_id) {
        // bh.print();
    }
    #endif

    #ifdef RUN_TEST
    bh.insert(tid);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) bhSeqGPU, bhSeqCPU);
}
