
#include "sll.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    Object obj1(1);
    Object obj2(2);
    Object obj3(3);

    Object objectPool[NUM_INSTANCES];
    SLLNode nodePool[NUM_INSTANCES];

    Env env(objectPool, nodePool);

    SLL sll(&env);

    sll.insertBack(&obj1);
    sll.insertBack(&obj2);
    sll.insertBack(&obj3);
    sll.print();
    sll.remove(0);
    sll.print();

    assert(!sll.contains(&obj1));
    assert(sll.contains(&obj2));
    return 0;
}
