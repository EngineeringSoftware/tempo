
#include "rbt.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    Object objectPool[NUM_INSTANCES];
    node nodePool[NUM_INSTANCES];
    Env env(objectPool, nodePool);
    int_red_black_tree rbt(&env);

    Object* o1 = new Object(4);
    Object* o2 = new Object(2);
    Object* o3 = new Object(6);
    Object* o4 = new Object(5);
    Object* o5 = new Object(7);
    Object* o6 = new Object(1);
    Object* o7 = new Object(3);

    rbt.put(4, o1);
    rbt.put(2, o2);
    rbt.put(6, o3);
    rbt.put(5, o4);
    rbt.put(7, o5);
    rbt.put(1, o6);
    rbt.put(3, o7);
    rbt.remove(4);

    assert(rbt.containsKey(5));
    assert(!rbt.containsKey(4));
    return 0;
}
