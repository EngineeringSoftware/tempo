
/*
 * This is C++ version of
 * predicate.taco.cachelist.NodeCachingLinkedList_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef CL_H_
#define CL_H_

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

class node_caching_linked_list {
private:
    __host__ __device__ void superRemoveNode(linked_list_node* node);
    __host__ __device__ int mathMin(int left, int right);
    __host__ __device__ linked_list_node* superCreateNode(Object* value);
    __host__ __device__ void superRemoveAllNodes();

protected:
    int size;
    int modCount;
    const int DEFAULT_MAXIMUM_CACHE_SIZE;
    linked_list_node* firstCachedNode;
    int cacheSize;
    int maximumCacheSize;
    Env *env;
    
    __host__ __device__ bool isEqualValue(const Object& value_1, const Object& value_2);
    __host__ __device__ int getMaximumCacheSize();
    __host__ __device__ void setMaximumCacheSize(int maximum_cache_size);
    __host__ __device__ void shrinkCacheToMaximumSize();
    __host__ __device__ linked_list_node* getNodeFromCache();
    __host__ __device__ bool isCacheFull();
    __host__ __device__ void addNodeToCache(linked_list_node* node);
    __host__ __device__ linked_list_node* createHeaderNode();
    __host__ __device__ linked_list_node* createNode(Object* value);
    __host__ __device__ void removeNode(linked_list_node* node);
    __host__ __device__ void removeAllNodes();
    __host__ __device__ void addNodeBefore(linked_list_node* node, Object* value);
    __host__ __device__ void addNode(linked_list_node* node_to_insert, linked_list_node* insert_before_node);
    __host__ __device__ linked_list_node* getNode(int index, bool end_marker_allowed);

public:
    linked_list_node* header;

    __host__ __device__ node_caching_linked_list(Env *env);

    __host__ __device__ bool remove(const Object& value);
    __host__ __device__ bool add(Object* value);
    __host__ __device__ bool addLast(Object* value);
    __host__ __device__ bool contains(const Object& value);
    __host__ __device__ int indexOf(const Object& value);
    __host__ __device__ Object* removeIndex(int index);

};

#endif
