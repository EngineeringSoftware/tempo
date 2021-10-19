
/*
 * This is C++ version of
 * predicate.issta2006.bintree.BinTree
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef BT_H_
#define BT_H_

class Node;

class Env {
public:
    /* pool of nodes to "allocate" from */
    Node *nodes;
    /* index of the next node to "allocate" */
    int8_t nodesIndex;

    __host__ __device__ Env(Node *nodes) : nodes(nodes), nodesIndex(0) {}

    __host__ __device__ Node *nodeAlloc();
};

class Node {
private:
    int value;
    Node *left;
    Node *right;
public:
    __device__ __host__ Node() : value(0), left(nullptr), right(nullptr) {}
    __device__ __host__ Node(int x) : value(x), left(nullptr), right(nullptr) {}

    __device__ __host__ int getValue() { return value; }
    __device__ __host__ Node* getLeft() { return left; }
    __device__ __host__ Node* getRight() { return right; }
    __device__ __host__ void setValue(int x) { value = x; }
    __device__ __host__ void setLeft(Node *n) { left = n; }
    __device__ __host__ void setRight(Node *n) { right = n; }
    __device__ __host__ void print();
};

class BT {
private:    
    Node *root;
    Env *env;
public:
    __device__ __host__ BT(Env *env) : env(env), root(nullptr) {}
    
    __device__ __host__ void add(int x);
    __device__ __host__ bool remove(int x);
    __device__ __host__ void print();
};

#endif
