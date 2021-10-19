
/*
 * This is C++ version of
 * predicate.issta2006.binomialheap.BinomialHeap_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef BH_H_
#define BH_H_

class binomial_heap_node;

class Env {
public:
    binomial_heap_node *nodePool;
    int8_t nodePoolIndex;

    __host__ __device__ Env(binomial_heap_node *nodePool) : nodePool(nodePool), nodePoolIndex(0) {}

    __host__ __device__ binomial_heap_node *nodeAlloc();
};

class binomial_heap_node {
public:
    int key;
    int degree;
    binomial_heap_node* parent;
    binomial_heap_node* sibling;
    binomial_heap_node* child;

    __host__ __device__ void setDegree(int deg) {
        degree = deg;
    }
    __host__ __device__ void setParent(binomial_heap_node* par) {
        parent = par;
    }
    __host__ __device__ void setSibling(binomial_heap_node* nextBr) {
        sibling = nextBr;
    }
    __host__ __device__ void setChild(binomial_heap_node* firstCh) {
        child = firstCh;
    }
    __host__ __device__ binomial_heap_node* reverse(binomial_heap_node* sibl) {
        binomial_heap_node* ret;
        if (sibling != nullptr) {
            ret = sibling->reverse(this);
        } else {
            ret = this;
        }

        sibling = sibl;
        return ret;
    }
    __host__ __device__ binomial_heap_node* findMinNode() {
        binomial_heap_node* x = this, *y = this;
        int min = x->key;
        while (x != nullptr) {
            if (x->key < min) {
                y = x;
                min = x->key;
            }

            x = x->sibling;
        }

        return y;
    }

    __host__ __device__ binomial_heap_node* findANodeWithKey(int value) {
        binomial_heap_node* temp = this, *node = nullptr;
        while (temp != nullptr) {
            if (temp->key == value) {
                node = temp;
                break;
            }

            if (temp->child == nullptr) {
                temp = temp->sibling;
            } else {
                node = temp->child->findANodeWithKey(value);
                if (node == nullptr) {
                    temp = temp->sibling;
                } else {
                    break;
                }
            }
        }

        return node;
    }


public:
    __host__ __device__ binomial_heap_node() : key(0), degree(0), parent(nullptr), sibling(nullptr), child(nullptr) {}
    __host__ __device__ binomial_heap_node(int k) : key(k), degree(0), parent(nullptr), sibling(nullptr), child(nullptr) {}

    __host__ __device__ void setKey(int value) {
        key = value;
    }
    __host__ __device__ int getKey() {
        return key;
    }
    __host__ __device__ int getDegree() {
        return degree;
    }
    __host__ __device__ binomial_heap_node* getParent() {
        return parent;
    }
    __host__ __device__ binomial_heap_node* getSibling() {
        return sibling;
    }
    __host__ __device__ binomial_heap_node* getChild() {
        return child;
    }
    __host__ __device__ int getSize() {
        return (1 + ((child == nullptr) ? 0 : child->getSize()) + ((sibling == nullptr) ? 0
                : sibling->getSize()));
    }
};

class binomial_heap {
private:
    binomial_heap_node* Nodes;
    int size;
    Env *env;

    __host__ __device__ void merge(binomial_heap_node* binHeap);
    __host__ __device__ void unionNodes(binomial_heap_node* binHeap);


public:
    __host__ __device__ binomial_heap(Env *env);

    __host__ __device__ int getSize();
    __host__ __device__ int findMinimum();
    __host__ __device__ void insert(int value);
    __host__ __device__ int extractMin();
    __host__ __device__ void decreaseKeyValue(int old_value, int new_value);
    __host__ __device__ void deleteNode(int value);

};

#endif
