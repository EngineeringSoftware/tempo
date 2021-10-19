#include <stdio.h>

#include "fh.h"
#include "../consts.h"

fibonacci_heap_node* Env::nodeAlloc() {
    fibonacci_heap_node *new_node = &(nodePool[nodePoolIndex]);
    nodePoolIndex++;
    if (nodePoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodePool, Index %d\n", nodePoolIndex);
    }

    return new_node;
}

fibonacci_heap::fibonacci_heap() : minNode(nullptr), size(0) {
}

void fibonacci_heap::deleteNode(fibonacci_heap_node* x) {
    // make x as small as possible
    decreaseKey(x, -2147483648);

    // remove the smallest, which decreases n also
    removeMin();
}

void fibonacci_heap::decreaseKey(fibonacci_heap_node* x, int k) {
    if (k > x->key) {
        return;
        // throw new IllegalArgumentException(
                // "decreaseKey() got larger key value");
    }

    x->key = k;
    fibonacci_heap_node* y = x->parent;
    if ((y != nullptr) && (x->key < y->key)) {
        cut(x, y);
        cascadingCut(y);
    }

    if (x->key < minNode->key) {
        minNode = x;
    }
}

void fibonacci_heap::cascadingCut(fibonacci_heap_node* y) {
    fibonacci_heap_node* z = y->parent;
    if (z != nullptr) {
        if (y->mark == FALSE) {
            y->mark = TRUE;
        } else {
            cut(y, z);
            cascadingCut(z);
        }
    }

}

void fibonacci_heap::cut(fibonacci_heap_node* x, fibonacci_heap_node* y) {
    x->left->right = x->right;
    x->right->left = x->left;
    y->degree--;
    if (y->child == x) {
        y->child = x->right;
    }

    if (y->degree == 0) {
        y->child = nullptr;
    }

    x->left = minNode;
    x->right = minNode->right;
    minNode->right = x;
    x->right->left = x;
    x->parent = nullptr;
    x->mark = FALSE;
}

fibonacci_heap_node* fibonacci_heap::removeMin() {
    fibonacci_heap_node* z = minNode;
    if (z != nullptr) {
        int numKids = z->degree;
        fibonacci_heap_node* x = z->child;
        fibonacci_heap_node* tempRight;
        while (numKids > 0) {
            tempRight = x->right;
            x->left->right = x->right;
            x->right->left = x->left;
            x->left = minNode;
            x->right = minNode->right;
            minNode->right = x;
            x->right->left = x;
            x->parent = nullptr;
            x = tempRight;
            numKids--;
        }
        z->left->right = z->right;
        z->right->left = z->left;
        if (z == z->right) {
            minNode = nullptr;
        } else {
            minNode = z->right;
            consolidate();
        }

        size--;
    }

    return z;
}

void fibonacci_heap::consolidate() {
    int arraySize = ((int) floor(log((double) size) * one_over_log_phi)) + 1;
    fibonacci_heap_node** array = new fibonacci_heap_node*[arraySize];
    // List<fibonacci_heap_node> array = new ArrayList<fibonacci_heap_node>(
            // arraySize);
    for (int i = 0; i < arraySize; i++) {
        array[i] = nullptr;
    }

    int numRoots = 0;
    fibonacci_heap_node* x = minNode;
    if (x != nullptr) {
        numRoots++;
        x = x->right;
        while (x != minNode) {
            numRoots++;
            x = x->right;
        }
    }

    while (numRoots > 0) {
        int d = x->degree;
        fibonacci_heap_node* next = x->right;
        for (;;) {
            fibonacci_heap_node* y = array[d];
            if (y == nullptr) {
                break;
            }

            if (x->key > y->key) {
                fibonacci_heap_node* temp = y;
                y = x;
                x = temp;
            }

            link(y, x);
            array[d] = nullptr;
            d++;
        }

        array[d] = x;
        x = next;
        numRoots--;
    }
    minNode = nullptr;
    for (int i = 0; i < arraySize; i++) {
        fibonacci_heap_node* y = array[i];
        if (y == nullptr) {
            continue;
        }

        if (minNode != nullptr) {
            y->left->right = y->right;
            y->right->left = y->left;
            y->left = minNode;
            y->right = minNode->right;
            minNode->right = y;
            y->right->left = y;
            if (y->key < minNode->key) {
                minNode = y;
            }
        } else {
            minNode = y;
        }
    }

    delete[] array;
}

void fibonacci_heap::link(fibonacci_heap_node* y, fibonacci_heap_node* x) {
    y->left->right = y->right;
    y->right->left = y->left;
    y->parent = x;
    if (x->child == nullptr) {
        x->child = y;
        y->right = y;
        y->left = y;
    } else {
        y->left = x->child;
        y->right = x->child->right;
        x->child->right = y;
        y->right->left = y;
    }

    x->degree++;
    y->mark = FALSE;
}

void fibonacci_heap::insert(fibonacci_heap_node* node, int key) {
    node->key = key;
    if (minNode != nullptr) {
        node->left = minNode;
        node->right = minNode->right;
        minNode->right = node;
        node->right->left = node;
        if (key < minNode->key) {
            minNode = node;
        }
    } else {
        minNode = node;
    }

    size++;
}

int fibonacci_heap::getMin() {
    fibonacci_heap_node* temp = minNode;
    int min = minNode->key;
    do {
        if (temp->key < min) {
            min = temp->key;
        }

        temp = temp->right;
    } while (temp != minNode);
    return min;
}

bool fibonacci_heap::checkHeapified() {
    fibonacci_heap_node* current = minNode;
    do {
        if (!current->checkHeapified()) {
            return false;
        }

        current = current->right;
    } while (current != minNode);
    return true;
}
