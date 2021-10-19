
#include "ts.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    tree_set_entry nodePool[NUM_INSTANCES];
    Env env(nodePool);
    tree_set ts(&env);

    ts.add(4);
    ts.add(2);
    ts.add(6);
    ts.add(5);
    ts.add(7);
    ts.add(1);
    ts.add(3);
    ts.remove(4);

    assert(ts.contains(2));
    assert(ts.contains(6));
    assert(ts.contains(5));
    assert(ts.contains(7));
    assert(ts.contains(1));
    assert(ts.contains(3));
    return 0;
}
