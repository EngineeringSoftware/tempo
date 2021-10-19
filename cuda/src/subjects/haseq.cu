
#include "ha.h"
#include "../mains.h"

__host__ void haSeqCPU(MethodSeq *ms) {
    int array[10];
    HA ha(array);
    for (int i = 0; i < msSize(ms); ++i) {
        switch (ms->ops[i]) {
        case 0:
            ha.insert(ms->vals[i]);
            break;
        case 1:
            ha.remove();
            break;
        }
    }
}

/* @private */
__global__ void haSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    int array[10];
    HA ha(array);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        switch (op) {
        case 0:
            ha.insert(value);
            break;
        case 1:
            ha.remove();
            break;
        }
    }

    _countIf(1);

    #ifdef HASEQ_DEBUG
    if (tid == print_id) {
        // ha.print();
    }
    #endif

    #ifdef RUN_TEST
    ha.insert(tid);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) haSeqGPU, haSeqCPU);
}
