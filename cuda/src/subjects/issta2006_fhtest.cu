
#include "issta2006_fh.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    node nodePool[NUM_INSTANCES];
    Env env(nodePool);
    fib_heap fh(&env);

    fh.insert(4);
    fh.insert(2);
    fh.insert(6);
    fh.insert(5);
    fh.insert(7);
    fh.insert(1);
    fh.insert(3);
    fh.removeMin();

    assert(fh.size() == 6);
    return 0;
}
