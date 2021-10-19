
#include "tm.h"
#include "../mains.h"

/* @private */
__host__ void tmSeqCPU(MethodSeq *ms) {
    entry nodeHeap[POOL_SIZE];
    Env env(nodeHeap);
    
    tree_map tm(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            tm.put(ms->vals[i]);
        } else {
            tm.remove(ms->vals[i]);
        }
    }
}

/* @private */
__global__ void tmSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    entry nodeHeap[POOL_SIZE];

    Env env(nodeHeap);
    
    tree_map tm(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            tm.put(value);
        } else {
            tm.remove(value);
        }
    }

    _countIf(1);

    #ifdef TMDSEQ_DEBUG
    if (tid == print_id) {
        // tm.print();
    }
    #endif

    #ifdef RUN_TEST
    tm.put(tid);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) tmSeqGPU, tmSeqCPU);
}
