
/*
 * This is C++ version of
 * predicate.taco.treeset.TreeSet_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef TS_H_
#define TS_H_

class tree_set_entry;

class Env {
public:
    tree_set_entry *nodePool;
    int8_t nodePoolIndex;

    __host__ __device__ Env(tree_set_entry *nodePool) : nodePool(nodePool), nodePoolIndex(0) {}

    __host__ __device__ tree_set_entry *nodeAlloc();
};

class tree_set_entry {
public:
    int key;
    tree_set_entry* left;
    tree_set_entry* right;
    tree_set_entry* parent;
    bool color;

    __host__ __device__ tree_set_entry()
        : key(0), left(nullptr), right(nullptr), parent(nullptr), color(true) {}

    __host__ __device__ tree_set_entry(int key, tree_set_entry* parent)
        : key(key), left(nullptr), right(nullptr), parent(parent), color(true) {}

    __host__ __device__ int getKey() { return key; }

    __host__ __device__ int hashCode() { return key; }
};

class tree_set {
private:
    tree_set_entry* root;
    int size;
    int mod_count;
    const bool RED;
    const bool BLACK;
    Env *env;

    __host__ __device__ tree_set_entry* getEntry(int key);
    __host__ __device__ void incrementSize();
    __host__ __device__ static bool colorOf(tree_set_entry* p);
    __host__ __device__ static tree_set_entry* parentOf(tree_set_entry* p);
    __host__ __device__ void setColor(tree_set_entry* p, bool c);
    __host__ __device__ static tree_set_entry* leftOf(tree_set_entry* p);
    __host__ __device__ static tree_set_entry* rightOf(tree_set_entry* p);
    __host__ __device__ void rotateLeft(tree_set_entry* p);
    __host__ __device__ void rotateRight(tree_set_entry* p);
    __host__ __device__ void fixAfterInsertion(tree_set_entry* entry);
    __host__ __device__ void deleteEntry(tree_set_entry* p);
    __host__ __device__ void fixAfterDeletion(tree_set_entry* p);
    __host__ __device__ void decrementSize();
    __host__ __device__ tree_set_entry* successor(tree_set_entry* t); 

public:
    __host__ __device__ tree_set(Env *env);

    __host__ __device__ bool contains(int a_key);
    __host__ __device__ bool add(int a_key);
    __host__ __device__ bool remove(int a_key);
};

#endif
