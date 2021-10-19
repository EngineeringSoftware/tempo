#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "sll.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void sllSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    Object objectHeap[POOL_SIZE];
    SLLNode nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    SLL sll(&env);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        Object* new_object = env.objectAlloc();
        if (op == 0) {
            // TODO: when we do want to free the objects?
            sll.insertBack(new_object);
        } else {
            // TODO: do we want to use value or someting else? in remove the
            // argument is an index
            sll.remove(value);
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))sllSeqOMP);
}
