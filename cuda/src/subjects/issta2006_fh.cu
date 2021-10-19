#include <stdio.h>

#include "issta2006_fh.h"
#include "../consts.h"

node* Env::nodeAlloc() {
    node *new_node = &(nodePool[nodePoolIndex]);
    nodePoolIndex++;
    if (nodePoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodePool, Index %d\n", nodePoolIndex);
    }

    return new_node;
}

fib_heap::fib_heap(Env* env)
    : min(nullptr), n(0), env(env) {
}

void fib_heap::cascadingCut(node* y) {
    node* z = y->parent;
    if (z != nullptr) {
        if (!y->mark) {
            y->mark = true;
        } else {
            cut(y, z);
            cascadingCut(z);
        }
    }
}

void fib_heap::consolidate() {
    int D = n + 1;
    node* A[10];
    // node** A = new node*[D];
    for (int i = 0; i < D; i++) {
        A[i] = nullptr;
    }

    int k = 0;
    node* x = min;
    if (x != nullptr) {
        k++;
        for (x = x->right; x != min; x = x->right) {
            k++;
        }
    }

    while (k > 0) {
        int d = x->degree;
        node* rightNode = x->right;
        while (A[d] != nullptr) {
            node* y = A[d];
            if (x->cost > y->cost) {
                node* temp = y;
                y = x;
                x = temp;
            }

            link(y, x);
            A[d] = nullptr;
            d++;
        }
        A[d] = x;
        x = rightNode;
        k--;
    }
    min = nullptr;
    for (int i = 0; i < D; i++) {
        if (A[i] != nullptr) {
            if (min != nullptr) {
                A[i]->left->right = A[i]->right;
                A[i]->right->left = A[i]->left;
                A[i]->left = min;
                A[i]->right = min->right;
                min->right = A[i];
                A[i]->right->left = A[i];
                if (A[i]->cost < min->cost) {
                    min = A[i];
                }
            } else {
                min = A[i];
            }
        }
    }
    // delete[] A;
}

void fib_heap::cut(node* x, node* y) {
    x->left->right = x->right;
    x->right->left = x->left;
    y->degree--;
    if (y->child == x) {
        y->child = x->right;
    }

    if (y->degree == 0) {
        y->child = nullptr;
    }

    x->left = min;
    x->right = min->right;
    min->right = x;
    x->right->left = x;
    x->parent = nullptr;
    x->mark = false;
}

void fib_heap::decreaseKey(node* x, int c) {
    if (c > x->cost) {
        printf("Error: new key is greater than current key->\n");
        return;
    }

    x->cost = c;
    node* y = x->parent;
    if ((y != nullptr) && (x->cost < y->cost)) {
        cut(x, y);
        cascadingCut(y);
    }

    if (x->cost < min->cost) {
        min = x;
    }

}

void fib_heap::deleteNode(node* n) {
    decreaseKey(n, -2147483648);
    removeMin();
    n->parent = nullptr;
    n->left = nullptr;
    n->right = nullptr;
    n->child = nullptr;
    delete n;
}

bool fib_heap::empty() {
    return min == nullptr;
}

void fib_heap::insert(int c) {
    node* n = env->nodeAlloc();
    n->cost = c;
    insert(n);
}

node* fib_heap::insert(node* toInsert) {
    if (min != nullptr) {
        toInsert->left = min;
        toInsert->right = min->right;
        min->right = toInsert;
        toInsert->right->left = toInsert;
        if (toInsert->cost < min->cost) {
            min = toInsert;
        }
    } else {
        min = toInsert;
    }

    n++;
    return toInsert;
}

void fib_heap::link(node* node1, node* node2) {
    node1->left->right = node1->right;
    node1->right->left = node1->left;
    node1->parent = node2;
    if (node2->child == nullptr) {
        node2->child = node1;
        node1->right = node1;
        node1->left = node1;
    } else {
        node1->left = node2->child;
        node1->right = node2->child->right;
        node2->child->right = node1;
        node1->right->left = node1;
    }

    node2->degree++;
    node1->mark = false;
}

node* fib_heap::minNode() {
    return min;
}

node* fib_heap::removeMin() {
    node* z = min;
    if (z != nullptr) {
        int i = z->degree;
        node* x = z->child;
        while (i > 0) {
            node* nextChild = x->right;
            x->left->right = x->right;
            x->right->left = x->left;
            x->left = min;
            x->right = min->right;
            min->right = x;
            x->right->left = x;
            x->parent = nullptr;
            x = nextChild;
            i--;
        }
        z->left->right = z->right;
        z->right->left = z->left;
        if (z == z->right) {
            min = nullptr;
        } else {
            min = z->right;
            consolidate();
        }

        n--;
    }

    return z;
}

int fib_heap::size() {
    return n;
}

fib_heap* fib_heap::heapUnion(fib_heap* heap1, fib_heap* heap2) {
    fib_heap* heap = new fib_heap(env);
    if ((heap1 != nullptr) && (heap2 != nullptr)) {
        heap->min = heap1->min;
        if (heap->min != nullptr) {
            if (heap2->min != nullptr) {
                heap->min->right->left = heap2->min->left;
                heap2->min->left->right = heap->min->right;
                heap->min->right = heap2->min;
                heap2->min->left = heap->min;
                if (heap2->min->cost < heap1->min->cost) {
                    heap->min = heap2->min;
                }
            }
        } else {
            heap->min = heap2->min;
        }

        heap->n = heap1->n + heap2->n;
    }

    return heap;
}
