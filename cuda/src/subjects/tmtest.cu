
#include "tm.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    entry nodePool[NUM_INSTANCES];
    Env env(nodePool);
    tree_map tm(&env);

    tm.put(4);
    tm.put(2);
    tm.put(6);
    tm.put(5);
    tm.put(7);
    tm.put(1);
    tm.put(3);
    tm.remove(4);

    assert(tm.getSize() == 6);
    assert(tm.containsKey(2));
    assert(tm.containsKey(6));
    assert(tm.containsKey(5));
    assert(tm.containsKey(7));
    assert(tm.containsKey(1));
    assert(tm.containsKey(3));

    return 0;
}
