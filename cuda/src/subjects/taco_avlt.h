
/*
 * This is C++ version of
 * predicate.taco.avltree.AvlTree_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef TACO_AVLT_H_
#define TACO_AVLT_H_

class avl_node;

class Env {
public:
    avl_node *nodePool;
    int8_t nodePoolIndex;

    __host__ __device__ Env(avl_node *nodePool) : nodePool(nodePool), nodePoolIndex(0) {}

    __host__ __device__ avl_node *nodeAlloc();
};

class avl_node {
public:
    int element;
    avl_node* left;
    avl_node* right;
    int height;

    __host__ __device__ avl_node() : avl_node(0, nullptr, nullptr) {}
    __host__ __device__ avl_node(int the_element) : avl_node(the_element, nullptr, nullptr) {}
    __host__ __device__ avl_node(int the_element, avl_node* lt, avl_node* rt)
        : element(the_element), left(lt), right(rt), height(0) {}
};

class avl_tree {
private:
    avl_node* root;
    Env *env;

    __host__ __device__ static avl_node* doubleWithLeftChild(avl_node* k3);
    __host__ __device__ static avl_node* doubleWithRightChild(avl_node* k1);
    __host__ __device__ static int height(avl_node* t);
    __host__ __device__ static int max(int lhs, int rhs);
    __host__ __device__ static avl_node* rotateWithLeftChild(avl_node* k2);
    __host__ __device__ static avl_node* rotateWithRightChild(avl_node* k1);
    __host__ __device__ bool balanced(avl_node* an);
    __host__ __device__ int elementAt(avl_node* t);
    __host__ __device__ avl_node* find(int x, avl_node* arg);
    __host__ __device__ avl_node* findMax(avl_node* arg);
    __host__ __device__ avl_node* findMin(avl_node* t);
    __host__ __device__ avl_node* insert(int x, avl_node* arg);
    __host__ __device__ avl_node* insert_0(int x, avl_node* arg);
    __host__ __device__ avl_node* insert_1(int x, avl_node* arg);
    __host__ __device__ avl_node* insert_2(int x, avl_node* arg);
    __host__ __device__ avl_node* insert_3(int x, avl_node* arg);
    __host__ __device__ avl_node* insert_4(int x, avl_node* arg);
    __host__ __device__ bool maxElement(int max, avl_node* t);
    __host__ __device__ bool minElement(int min, avl_node* t);
    __host__ __device__ int mathMax(int l, int r);
    __host__ __device__ bool wellFormed(avl_node* an);

public:
    __host__ __device__ avl_tree(Env *env);

    __host__ __device__ bool balanced();
    __host__ __device__ int find(int x);
    __host__ __device__ avl_node* findNode(int x);
    __host__ __device__ int findMax();
    __host__ __device__ avl_node* fmax();
    __host__ __device__ int findMin();
    __host__ __device__ void insertElem(int x);
    __host__ __device__ bool isEmpty();
    __host__ __device__ void makeEmpty();
    __host__ __device__ bool maxElement(int max);
    __host__ __device__ bool minElement(int min);
    __host__ __device__ void remove(int x);
    __host__ __device__ bool wellFormed();
};

#endif
