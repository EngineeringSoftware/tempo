
/*
 * This is C++ version of
 * predicate.fibonacciheap.FibonacciHeap_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef FH_H_
#define FH_H_

class fibonacci_heap_node;

class Env {
public:
    fibonacci_heap_node *nodePool;
    int8_t nodePoolIndex;

    __host__ __device__ Env(fibonacci_heap_node *nodePool) : nodePool(nodePool), nodePoolIndex(0) {}

    __host__ __device__ fibonacci_heap_node *nodeAlloc();
};

class fibonacci_heap_node {
public:
    int key;
    int degree;
    fibonacci_heap_node* parent;
    fibonacci_heap_node* child;
    fibonacci_heap_node* left;
    fibonacci_heap_node* right;
    int mark;

    __host__ __device__ fibonacci_heap_node() : key(0), degree(0), parent(nullptr),
                            child(nullptr) , left(this), right(this), mark(0) {}

    __host__ __device__ int getSize(fibonacci_heap_node* fib_node) {
        int result = 1;
        if (child != nullptr)
            result += child->getSize(child);
        if (right != fib_node)
            result += right->getSize(fib_node);
        return result;
    }

    __host__ __device__ bool contains(fibonacci_heap_node* start, fibonacci_heap_node* node) {
        fibonacci_heap_node* temp = start;
        do {
            if (temp == node)
                return true;
            else
                temp = temp->right;
        } while (temp != start);
        return false;
    }

    __host__ __device__ bool isEqualTo(fibonacci_heap_node* node) {
        fibonacci_heap_node* temp_this = this;
        fibonacci_heap_node* temp_that = node;
        do {
            if ((temp_this->key != temp_that->key)
                    || (temp_this->degree != temp_that->degree)
                    || (temp_this->mark != temp_that->mark)
                    || ((temp_this->child != nullptr) && (temp_that->child == nullptr))
                    || ((temp_this->child == nullptr) && (temp_that->child != nullptr))
                    || ((temp_this->child != nullptr) && (!temp_this->child
                            ->isEqualTo(temp_that->child))))
                return false;
            else {
                temp_this = temp_this->right;
                temp_that = temp_that->right;
            }
        } while (temp_this->right != this);
        return true;
    }

    __host__ __device__ fibonacci_heap_node* findKey(fibonacci_heap_node* start, int k) {
        fibonacci_heap_node* temp = start;
        do
            if (temp->key == k)
                return temp;
            else {
                fibonacci_heap_node* child_temp = nullptr;
                if ((temp->key < k) && (temp->child != nullptr))
                    child_temp = temp->child->findKey(temp->child, k);
                if (child_temp != nullptr)
                    return child_temp;
                else
                    temp = temp->right;
            }
        while (temp != start);
        return nullptr;
    }

    __host__ __device__ int numberOfChildren() {
        if (child == nullptr)
            return 0;
        int num = 1;
        for (fibonacci_heap_node* current = child->right; current != child; current = current->right) {
            num++;
        }
        return num;
    }

    __host__ __device__ bool checkDegrees() {
        fibonacci_heap_node* current = this;
        do {
            if (current->numberOfChildren() != current->degree)
                return false;
            if (current->child != nullptr)
                if (!current->child->checkDegrees())
                    return false;
            current = current->right;
        } while (current != this);
        return true;
    }

    __host__ __device__ bool checkHeapified() {
        touch(key);
        if (child == nullptr)
            return true;
        fibonacci_heap_node* current = child;
        do {
            if (current->key < key)
                return false;
            if (!current->checkHeapified())
                return false;
            current = current->right;
        } while (current != child);
        return true;
    }

    __host__ __device__ void touch(int key) {
    }
};

class fibonacci_heap {
private:
    const double one_over_log_phi = 1.0 / log((1.0 + sqrt(5.0)) / 2.0);
    const static int FALSE = 0;
    const static int TRUE = 1;

    __host__ __device__ int getMin();

protected:
    __host__ __device__ void cascadingCut(fibonacci_heap_node* y);
    __host__ __device__ void cut(fibonacci_heap_node* x, fibonacci_heap_node* y);
    __host__ __device__ void consolidate();
    __host__ __device__ void link(fibonacci_heap_node* y, fibonacci_heap_node* x);

public:
    fibonacci_heap_node* minNode;
    int size;

    __host__ __device__ fibonacci_heap();

    __host__ __device__ void deleteNode(fibonacci_heap_node* x);
    __host__ __device__ void decreaseKey(fibonacci_heap_node* x, int k);
    __host__ __device__ fibonacci_heap_node* removeMin();
    __host__ __device__ void insert(fibonacci_heap_node* node, int key);
    __host__ __device__ bool checkHeapified();

};

#endif
