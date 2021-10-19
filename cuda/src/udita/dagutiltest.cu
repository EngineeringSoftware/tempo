
#include <assert.h>
#include "dag.h"

__global__ void ll_test() {
    LinkedList ll;
    llInit(&ll);

    Node *n1 = (Node*) malloc(sizeof(Node));
    Node *n2 = (Node*) malloc(sizeof(Node));
    
    llAdd(&ll, n1);
    llAdd(&ll, n2);

    assert(ll.size == 2);

    Node *n3 = llRemoveFirst(&ll);
    
    assert(ll.size == 1);
    assert(n3 == n1);

    Node *n4 = (Node*) malloc(sizeof(Node));

    llAdd(&ll, n1);
    assert(ll.size == 2);

    llAdd(&ll, n4);
    assert(ll.size == 3);

    Node *n5 = llRemoveLast(&ll);
    assert(n5 == n4);
    assert(ll.size == 2);
}

__global__ void set_test() {
    LinkedList ll;
    Set s;
    setInit(&s, &ll);

    Node *n1 = (Node*) malloc(sizeof(Node));
    Node *n2 = (Node*) malloc(sizeof(Node));
    
    setAdd(&s, n1);
    setAdd(&s, n2);

    assert(setSize(&s) == 2);

    setAdd(&s, n1);
    setAdd(&s, n2);

    assert(setSize(&s) == 2);

    assert(setContains(&s, n1));
    assert(setContains(&s, n2));

    Node *n3 = (Node*) malloc(sizeof(Node));
    assert(!setContains(&s, n3));

    setAdd(&s, n3);
    assert(setSize(&s) == 3);
    assert(setContains(&s, n3));

    setRemove(&s, n2);
    assert(setSize(&s) == 2);
    assert(!setContains(&s, n2));

    Node *n4 = (Node*) malloc(sizeof(Node));
    assert(!setContains(&s, n4));

    setAdd(&s, n4);
    assert(setContains(&s, n4));
    assert(setSize(&s) == 3);

    setRemove(&s, n3);
    assert(setSize(&s) == 2);
    assert(!setContains(&s, n3));
}

int main(void) {
    set_test<<<1,1>>>();
    cudaDeviceSynchronize();

    ll_test<<<1,1>>>();
    cudaDeviceSynchronize();
}
