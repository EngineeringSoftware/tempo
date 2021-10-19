#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "cl.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void clSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    Object objectHeap[POOL_SIZE];
    linked_list_node nodeHeap[POOL_SIZE];

    Env env(objectHeap, nodeHeap);

    node_caching_linked_list cl(&env);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        if (op == 0) {
            Object* object = env.objectAlloc();
            object->id = value;
            cl.add(object);
        } else {
            cl.remove(Object(value));
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))clSeqOMP);
}
