
#include "issta2006_fh.h"
#include "../mains.h"
#include "../consts.h"

/* @private */
__host__ void issta2006_fhSeqCPU(MethodSeq *ms) {
    node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    fib_heap fh(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            fh.insert(ms->vals[i]);
        } else {
            fh.removeMin();
        }
    }
}

/* @private */
__global__ void issta2006_fhSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    fib_heap fh(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            fh.insert(value);
        } else {
            fh.removeMin();
        }
    }

    _countIf(1);

    #ifdef ISSTA2006_FHSEQ_DEBUG
    if (tid == print_id) {
        // fh.print();
    }
    #endif

    #ifdef RUN_TEST
    fh.insert(tid);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) issta2006_fhSeqGPU, issta2006_fhSeqCPU);
}
