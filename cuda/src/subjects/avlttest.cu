
#include "avlt.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    Object objectHeap[NUM_INSTANCES];
    Node nodeHeap[NUM_INSTANCES];
    Env env(objectHeap, nodeHeap);

    int_avl_tree_map avl(&env);

    Object* o1 = new Object(4);
    Object* o2 = new Object(2);
    Object* o3 = new Object(6);
    Object* o4 = new Object(5);
    Object* o5 = new Object(7);
    Object* o6 = new Object(1);
    Object* o7 = new Object(3);

    avl.put(4, o1);
    avl.put(2, o2);
    avl.put(6, o3);
    avl.put(5, o4);
    avl.put(7, o5);
    avl.put(1, o6);
    avl.put(3, o7);
    avl.remove(4);

    assert(avl.getSize() == 6);
    assert(avl.containsKey(2));
    assert(avl.containsKey(6));
    assert(avl.containsKey(5));
    assert(avl.containsKey(7));
    assert(avl.containsKey(1));
    assert(avl.containsKey(3));

    return 0;
}
