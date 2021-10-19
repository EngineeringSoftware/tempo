
/*
 * This is C++ version of
 * predicate.avl.IntAVLTreeMap_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef AVLT_H_
#define AVLT_H_

#include <stdio.h>

class Object;
class Node;

class Env {
public:
    Object *objectHeap;
    Node *nodeHeap;
    int8_t objectsIndex;
    int8_t nodesIndex;

    __host__ __device__ Env(Object *objectHeap, Node *nodeHeap) : objectHeap(objectHeap), nodeHeap(nodeHeap), objectsIndex(0), nodesIndex(0) {}

    __host__ __device__ Node *nodeAlloc();
    __host__ __device__ Object *objectAlloc();
};

/* Fake class to get closer to Java implementation */
class Object {
public:
    int id;
    __host__ __device__ Object(int val): id(val) {}
    __host__ __device__ void print() { printf("%d", id); }
    __host__ __device__ Object() {}
};

class Node {
public:
    int key;
    Object* value;
    Node* left;
    Node* right;
    int height;

    __host__ __device__ Node() : value(nullptr), left(nullptr), right(nullptr) {}
};

class int_avl_tree_map {
private:
    Node *root;
    int size;
    Env *env;

    __host__ __device__ int getHeight(Node* x);
    __host__ __device__ int getBalance(Node* x);
    __host__ __device__ Node* findNode(int key);
    __host__ __device__ Node* rightRotate(Node* x);
    __host__ __device__ Node* leftRotate(Node* x);
    __host__ __device__ Node* rightLeftRotate(Node *x);
    __host__ __device__ Node* leftRightRotate(Node *x);
    __host__ __device__ Node* treeInsertRecur(Node *x, Node *cur);
    __host__ __device__ Node* treeDelete(Node *x);
    __host__ __device__ Node* treeDeleteRecur(Node *x, Node *cur);
    __host__ __device__ Node* deleteFix(Node *cur);
    __host__ __device__ Node* afterDelete(Node *x);
    __host__ __device__ Node* getIOS(Node *z);

public:
    __host__ __device__ int_avl_tree_map(Env *env);

    __host__ __device__ bool containsKey(int key);
    __host__ __device__ Object* get(int key);
    __host__ __device__ void put(int key, Object* value);
    __host__ __device__ Object* remove(int key);
    __host__ __device__ int getSize();    
    __host__ __device__ void treeInsert(Node* x);
};

#endif
