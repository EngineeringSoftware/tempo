
/*
 * This is C++ version of
 * predicate.taco.intredblacktree.IntRedBlackTree_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef RBT_H_
#define RBT_H_

#include <stdio.h>

class Object;
class node;

class Env {
public:
    Object *objectPool;
    node *nodePool;
    int8_t objectPoolIndex;
    int8_t nodePoolIndex;

    __host__ __device__ Env(Object *objectPool, node *nodePool) : objectPool(objectPool), nodePool(nodePool), objectPoolIndex(0), nodePoolIndex(0) {}

    __host__ __device__ node *nodeAlloc();
    __host__ __device__ Object *objectAlloc();
};

/* Fake class to get closer to Java implementation */
class Object {
public:
    int id;
    __host__ __device__ Object(): id(0) {}
    __host__ __device__ Object(int val): id(val) {}
    __host__ __device__ void print() { printf("%d", id); }
};

class node {
public:
    int key;
    Object* value;
    node* parent;
    node* left;
    node* right;
    bool color;

    __host__ __device__ node() {}

};

class int_red_black_tree {
protected:
    const static bool RED = false;
    const static bool BLACK = true;

    node* root;
    int size;
    Env *env;

    __host__ __device__ node* parent(node* n);
    __host__ __device__ node* left(node* n);
    __host__ __device__ node* right(node* n);
    __host__ __device__ bool getColor(node* x);
    __host__ __device__ void setColor(node* x, bool color);
    __host__ __device__ node* findNode(int key);
    __host__ __device__ void leftRotate(node* x);
    __host__ __device__ void rightRotate(node* x);
    __host__ __device__ void treeInsert(node* z);
    __host__ __device__ void treeInsertFix(node* z);
    __host__ __device__ node* treeDelete(node* z);
    __host__ __device__ void treeDeleteFix(node* x);
    __host__ __device__ node* getIOS(node* z);

public:
    __host__ __device__ int_red_black_tree(Env *env);

    __host__ __device__ bool containsKey(int key);
    __host__ __device__ Object* get(int key);
    __host__ __device__ void put(int key, Object* value);
    __host__ __device__ Object* remove(int key);
};

#endif
