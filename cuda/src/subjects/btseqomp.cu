#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "bt.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void btSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    Node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);
    
    BT bt(&env);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        if (op == 0) {
            bt.add(value);
        } else {
            bt.remove(value);
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))btSeqOMP);
}
