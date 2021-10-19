#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "taco_avlt.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void taco_avltSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    avl_node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);
    
    avl_tree avl(&env);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        if (op == 0) {
            avl.insertElem(value);
        } else {
            avl.remove(value);
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))taco_avltSeqOMP);
}
