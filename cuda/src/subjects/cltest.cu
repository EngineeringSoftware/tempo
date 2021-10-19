
#include "cl.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    Object objectPool[NUM_INSTANCES];
    linked_list_node nodePool[NUM_INSTANCES];
    Env env(objectPool, nodePool);
    node_caching_linked_list cl(&env);

    Object* o1 = new Object(4);
    Object* o2 = new Object(2);
    Object* o3 = new Object(6);
    Object* o4 = new Object(5);
    Object* o5 = new Object(7);
    Object* o6 = new Object(1);
    Object* o7 = new Object(3);

    cl.add(o1);
    cl.add(o2);
    cl.add(o3);
    cl.add(o4);
    cl.add(o5);
    cl.add(o6);
    cl.add(o7);

    assert(cl.contains(*o1));
    assert(cl.contains(*o2));
    assert(cl.contains(*o3));
    assert(cl.contains(*o4));
    assert(cl.contains(*o5));
    assert(cl.contains(*o6));
    assert(cl.contains(*o7));

    return 0;
}
