#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "ts.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void tsSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    tree_set_entry nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    tree_set ts(&env);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        if (op == 0) {
            ts.add(value);
        } else {
            ts.remove(value);
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))tsSeqOMP);
}
