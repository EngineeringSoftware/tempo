
/*
 * This is C++ version of
 * predicate.taco.linkedlist.LinkedList_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef LL_H_
#define LL_H_

#include <stdio.h>

class Object;
struct linked_list_node;

class Env {
public:
    Object *objectPool;
    linked_list_node *nodePool;
    int8_t objectPoolIndex;
    int8_t nodePoolIndex;

    __host__ __device__ Env(Object *objectPool, linked_list_node *nodePool) : objectPool(objectPool), nodePool(nodePool), objectPoolIndex(0), nodePoolIndex(0) {}

    __host__ __device__ linked_list_node *nodeAlloc();
    __host__ __device__ Object *objectAlloc();
};

/* Fake class to get closer to Java implementation */
class Object {
public:
    int id;
    __host__ __device__ Object() : id(0) {}
    __host__ __device__ Object(int val): id(val) {}
    __host__ __device__ void print() { printf("%d", id); }
};

struct linked_list_node {
    linked_list_node* previous;
    linked_list_node* next;
    Object* value;

    __host__ __device__ linked_list_node() : linked_list_node(nullptr) {}
    __host__ __device__ linked_list_node(Object* value) : previous(this), next(this), value(value) {}
    __host__ __device__ linked_list_node(linked_list_node* previous, linked_list_node* next, Object* value)
        : previous(previous), next(next), value(value) {}

    __host__ __device__ void setValue(Object* value) {
        this->value = value;
    }

    __host__ __device__ Object* getValue() {
        return value;
    } 
};

class linked_list {
protected:
    int modCount;
    Env *env;

    __host__ __device__ bool isEqualValue(const Object& value_1, const Object& value_2);
    __host__ __device__ void updateNode(linked_list_node* node, Object* value);
    __host__ __device__ linked_list_node* createHeaderNode();
    __host__ __device__ linked_list_node* createNode(Object* value);
    __host__ __device__ void addNodeBefore(linked_list_node* node, Object* value);
    __host__ __device__ void addNodeAfter(linked_list_node* node, Object* value);
    __host__ __device__ void addNode(linked_list_node* node_to_insert, linked_list_node* insert_before_node);
    __host__ __device__ void removeNode(linked_list_node* node);
    __host__ __device__ void removeAllNodes();
    __host__ __device__ linked_list_node* getNode(int index, bool end_marker_allowed);
    
public:
    int size;
    linked_list_node* header;

    __host__ __device__ linked_list(Env *env);

    __host__ __device__ int getSize();
    __host__ __device__ bool isEmpty();
    __host__ __device__ Object* get(int index);
    __host__ __device__ int indexOf(const Object& value);
    __host__ __device__ int lastIndexOf(const Object& value);
    __host__ __device__ bool contains(const Object& value);
    __host__ __device__ bool add(Object* value);
    __host__ __device__ void add(int index, Object* value);
    __host__ __device__ Object* removeIndex(int index);
    __host__ __device__ bool remove(const Object& value);
    __host__ __device__ Object* set(int index, Object* value);
    __host__ __device__ void clear();
    __host__ __device__ Object* getFirst();
    __host__ __device__ Object* getLast();
    __host__ __device__ bool addFirst(Object* value);
    __host__ __device__ bool addLast(Object* value);
    __host__ __device__ Object* removeFirst();
    __host__ __device__ Object* removeLast();

};

#endif
