
#include "fh.h"
#include "../mains.h"

/* @private */
__host__ void fhSeqCPU(MethodSeq *ms) {
    fibonacci_heap_node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    fibonacci_heap fh;
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            fh.insert(env.nodeAlloc(), ms->vals[i]);
        } else {
            fh.removeMin();
        }
    }
}

/* @private */
__global__ void fhSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    fibonacci_heap_node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    fibonacci_heap fh;
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            fh.insert(env.nodeAlloc(), value);
        } else {
            fh.removeMin();
        }
    }

    _countIf(1);

    #ifdef FHSEQ_DEBUG
    if (tid == print_id) {
        // fh.print();
    }
    #endif

    #ifdef RUN_TEST
    fh.insert(env.nodeAlloc(), tid);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) fhSeqGPU, fhSeqCPU);
}
