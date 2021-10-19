
#include "ts.h"
#include "../mains.h"

/* @private */
__host__ void tsSeqCPU(MethodSeq *ms) {
    tree_set_entry nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    tree_set ts(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            ts.add(ms->vals[i]);
        } else {
            ts.remove(ms->vals[i]);
        }
    }
}

/* @private */
__global__ void tsSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    tree_set_entry nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    tree_set ts(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            ts.add(value);
        } else {
            ts.remove(value);
        }
    }

    _countIf(1);

    #ifdef TSSEQ_DEBUG
    if (tid == print_id) {
        // ts.print();
    }
    #endif

    #ifdef RUN_TEST
    ts.add(tid);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) tsSeqGPU, tsSeqCPU);
}
