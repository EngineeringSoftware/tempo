
/*
 * This is C++ version of
 * predicate.issta2006.treemap.TreeMap_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef TM_H_
#define TM_H_

class entry;

class Env {
public:
    entry *nodePool;
    int8_t nodePoolIndex;

    __host__ __device__ Env(entry *nodePool) : nodePool(nodePool), nodePoolIndex(0) {}

    __host__ __device__ entry *nodeAlloc();
};

class entry {
public:
    int key;
    entry* left = nullptr;
    entry* right = nullptr;
    entry* parent;
    bool color = true;

    __host__ __device__ entry() : key(0), left(nullptr), right(nullptr), parent(nullptr), color(true) {}
    __host__ __device__ entry(int key, entry* parent) : key(key), left(nullptr), right(nullptr), parent(parent), color(true) {}
    __host__ __device__ entry(int key, entry* left, entry* right, entry* parent, bool color)
        : key(key), left(nullptr), right(nullptr), parent(parent), color(true) {}

    __host__ __device__ int getKey() {
        return key;
    }
};

class tree_map {
private:
    entry* root;
    int size;
    const static bool RED = false;
    const static bool BLACK = true;
    Env *env;

    __host__ __device__ void incrementSize();
    __host__ __device__ void decrementSize();
    __host__ __device__ entry* getEntry(int key);
    __host__ __device__ entry* successor(entry* t);
    __host__ __device__ static bool colorOf(entry* p);
    __host__ __device__ static entry* parentOf(entry* p);
    __host__ __device__ static void setColor(entry* p, bool c);
    __host__ __device__ static entry* leftOf(entry* p);
    __host__ __device__ static entry* rightOf(entry* p);
    __host__ __device__ void rotateLeft(entry* p);
    __host__ __device__ void rotateRight(entry* p);
    __host__ __device__ void fixAfterInsertion(entry* x);
    __host__ __device__ void deleteEntry(entry* p);
    __host__ __device__ void fixAfterDeletion(entry* x);
    __host__ __device__ void swapPosition(entry* x, entry* y);

public:
    __host__ __device__ tree_map(Env *env);

    __host__ __device__ int getSize();
    __host__ __device__ bool containsKey(int key);
    __host__ __device__ void put(int key);
    __host__ __device__ void remove(int key);
};

#endif
