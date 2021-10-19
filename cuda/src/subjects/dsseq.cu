
#include "ds.h"
#include "../mains.h"

__host__ void dsSeqCPU(MethodSeq *ms) {
    // Record array[10];
    // disjset ds(array);
    // ds.setPathCompression(true);
    // ds.create(msSize(ms));
    // for (int i = 0; i < msSize(ms); ++i) {
    //     switch (msSize(ms)->ops[i]) {
    //         case 0:
    //         int el = _choice(0, n - 1);
    //         ds.find(el);
    //         break;
    //     case 1:
    //         int x = _choice(0, n - 1);
    //         int y = _choice(0, n - 1);
    //         _ignoreIf(x == y);
    //         ds.unionMethod(x,y);
    //         break;
    //     }
    // }
}

/* @private */
__global__ void dsSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }

    Record array[10];
    disjset ds(array);
    ds.setPathCompression(true);
    ds.create(n);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        switch (op) {
            case 0:
            const int el = _choice(0, n - 1);
            ds.find(el);
            break;
        case 1:
            const int x = _choice(0, n - 1);
            const int y = _choice(0, n - 1);
            _ignoreIf(x == y);
            ds.unionMethod(x,y);
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
    ds.find(tid % n);
    #endif
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) dsSeqGPU, dsSeqCPU);
}
