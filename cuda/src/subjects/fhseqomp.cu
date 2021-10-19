#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "fh.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void fhSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    fibonacci_heap_node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    fibonacci_heap fh;
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        if (op == 0) {
            fh.insert(env.nodeAlloc(), value);
        } else {
            fh.removeMin();
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))fhSeqOMP);
}
