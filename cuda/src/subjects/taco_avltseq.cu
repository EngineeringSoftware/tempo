
#include "taco_avlt.h"
#include "../mains.h"

/* @private */
__host__ void taco_avltSeqCPU(MethodSeq *ms) {
    avl_node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);
    
    avl_tree avl(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            avl.insertElem(ms->vals[i]);
        } else {
            avl.remove(ms->vals[i]);
        }
    }
}

/* @private */
__global__ void taco_avltSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    avl_node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);
    
    avl_tree avl(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            avl.insertElem(value);
        } else {
            avl.remove(value);
        }
    }

    _countIf(1);

    #ifdef TACO_AVLTSEQ_DEBUG
    if (tid == print_id) {
        // avl.print();
    }
    #endif

    #ifdef RUN_TEST
    avl.insertElem(tid);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) taco_avltSeqGPU, taco_avltSeqCPU);
}
