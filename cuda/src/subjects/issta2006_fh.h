
/*
 * This is C++ version of
 * predicate.issta2006.fibheap.FibHeap_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef ISSTA2006_FH_H_
#define ISSTA2006_FH_H_

class node;

class Env {
public:
    node *nodePool;
    int8_t nodePoolIndex;

    __host__ __device__ Env(node *nodePool) : nodePool(nodePool), nodePoolIndex(0) {}

    __host__ __device__ node *nodeAlloc();
};

class node {
public:
    node* parent;
    node* left;
    node* right;
    node* child;
    bool mark;
    int cost;
    int degree;

    __host__ __device__ node() : parent(nullptr), left(this), right(this), child(nullptr), mark(false), cost(0), degree(0) {};
    __host__ __device__ node(int c) : parent(nullptr), left(this), right(this), child(nullptr), mark(false), cost(c), degree(0) {}
};

class fib_heap {
private:
    node* min;
    int n;
    Env* env;

    __host__ __device__ void cascadingCut(node* y);
    __host__ __device__ void consolidate();
    __host__ __device__ void cut(node* x, node* y);
    __host__ __device__ void link(node* node1, node* node2);

public:
    __host__ __device__ fib_heap(Env* env);

    __host__ __device__ void decreaseKey(node* x, int c);
    __host__ __device__ void deleteNode(node* n);
    __host__ __device__ bool empty();
    __host__ __device__ void insert(int c);
    __host__ __device__ node* insert(node* toInsert);
    __host__ __device__ node* minNode();
    __host__ __device__ node* removeMin();
    __host__ __device__ int size();
    __host__ __device__ fib_heap* heapUnion(fib_heap* heap1, fib_heap* heap2);

};

#endif
