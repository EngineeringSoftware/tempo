
#include "fh.h"
#include <assert.h>

#define NUM_INSTANCES 100

int main(void) {
    fibonacci_heap_node nodePool[NUM_INSTANCES];
    Env env(nodePool);
    fibonacci_heap fh;

    fibonacci_heap_node* n1 = new fibonacci_heap_node();
    fibonacci_heap_node* n2 = new fibonacci_heap_node();
    fibonacci_heap_node* n3 = new fibonacci_heap_node();
    fibonacci_heap_node* n4 = new fibonacci_heap_node();
    fibonacci_heap_node* n5 = new fibonacci_heap_node();
    fibonacci_heap_node* n6 = new fibonacci_heap_node();
    fibonacci_heap_node* n7 = new fibonacci_heap_node();

    fh.insert(n1, 4);
    fh.insert(n2, 2);
    fh.insert(n3, 6);
    fh.insert(n4, 5);
    fh.insert(n5, 7);
    fh.insert(n6, 1);
    fh.insert(n7, 3);
    fh.removeMin();

    assert(fh.size == 6);
    return 0;
}
