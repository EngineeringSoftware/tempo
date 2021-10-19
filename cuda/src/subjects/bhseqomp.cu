#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "bh.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void bhSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    binomial_heap_node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    binomial_heap bh(&env);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        if (op == 0) {
            bh.insert(value);
        } else {
            bh.deleteNode(value);
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))bhSeqOMP);
}
