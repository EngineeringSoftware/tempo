
#include "ll.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    Object objectPool[NUM_INSTANCES];
    linked_list_node nodePool[NUM_INSTANCES];
    Env env(objectPool, nodePool);
    linked_list ll(&env);

    Object* o1 = new Object(4);
    Object* o2 = new Object(2);
    Object* o3 = new Object(6);
    Object* o4 = new Object(5);
    Object* o5 = new Object(7);
    Object* o6 = new Object(1);
    Object* o7 = new Object(3);

    ll.add(o1);
    ll.add(o2);
    ll.add(o3);
    ll.add(o4);
    ll.add(o5);
    ll.add(o6);
    ll.add(o7);
    ll.removeFirst();
    ll.removeLast();

    assert(ll.size == 6);
    return 0;
}
