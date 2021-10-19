
#include "bh.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    binomial_heap_node nodePool[NUM_INSTANCES];
    Env env(nodePool);
    binomial_heap bh(&env);

    bh.insert(4);
    bh.insert(2);
    bh.insert(6);
    bh.insert(5);
    bh.insert(7);
    bh.insert(1);
    bh.insert(3);
    bh.deleteNode(2);

    assert(bh.getSize() == 6);

    return 0;
}
