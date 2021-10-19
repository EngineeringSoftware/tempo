
#include "taco_avlt.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    avl_node nodePool[NUM_INSTANCES];
    Env env(nodePool);
    avl_tree avl(&env);

    avl.insertElem(4);
    avl.insertElem(2);
    avl.insertElem(6);
    avl.insertElem(5);
    avl.insertElem(7);
    avl.insertElem(1);
    avl.insertElem(3);
    avl.remove(4);

    assert(avl.findMax() == 7);
    return 0;
}
