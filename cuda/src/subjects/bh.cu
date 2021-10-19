#include <stdio.h>

#include "bh.h"
#include "../consts.h"

binomial_heap_node* Env::nodeAlloc() {
    binomial_heap_node *new_node = &(nodePool[nodePoolIndex]);
    nodePoolIndex++;
    if (nodePoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodePool, Index %d\n", nodePoolIndex);
    }

    return new_node;
}

binomial_heap::binomial_heap(Env *env) : Nodes(nullptr), size(0), env(env) {
}

int binomial_heap::getSize() {
    return size;
}

int binomial_heap::findMinimum() {
    return Nodes->findMinNode()->key;
}

void binomial_heap::merge(binomial_heap_node* binHeap) {
    binomial_heap_node* temp1 = Nodes, *temp2 = binHeap;
    while ((temp1 != nullptr) && (temp2 != nullptr)) {
        if (temp1->degree == temp2->degree) {
            binomial_heap_node* tmp = temp2;
            temp2 = temp2->sibling;
            tmp->sibling = temp1->sibling;
            temp1->sibling = tmp;
            temp1 = tmp->sibling;
        } else {
            if (temp1->degree < temp2->degree) {
                if ((temp1->sibling == nullptr)
                        || (temp1->sibling->degree > temp2->degree)) {
                    binomial_heap_node* tmp = temp2;
                    temp2 = temp2->sibling;
                    tmp->sibling = temp1->sibling;
                    temp1->sibling = tmp;
                    temp1 = tmp->sibling;
                } else {
                    temp1 = temp1->sibling;
                }
            } else {
                binomial_heap_node* tmp = temp1;
                temp1 = temp2;
                temp2 = temp2->sibling;
                temp1->sibling = tmp;
                if (tmp == Nodes) {
                    Nodes = temp1;
                }
            }
        }
    }
    if (temp1 == nullptr) {
        temp1 = Nodes;
        while (temp1->sibling != nullptr) {
            temp1 = temp1->sibling;
        }
        temp1->sibling = temp2;
    }
}

void binomial_heap::unionNodes(binomial_heap_node* binHeap) {
    merge(binHeap);
    binomial_heap_node* prevTemp = nullptr, *temp = Nodes, *nextTemp = Nodes->sibling;
    while (nextTemp != nullptr) {
        if ((temp->degree != nextTemp->degree)
                || ((nextTemp->sibling != nullptr) && (nextTemp->sibling->degree == temp->degree))) {
            prevTemp = temp;
            temp = nextTemp;
        } else {
            if (temp->key <= nextTemp->key) {
                temp->sibling = nextTemp->sibling;
                nextTemp->parent = temp;
                nextTemp->sibling = temp->child;
                temp->child = nextTemp;
                temp->degree++;
            } else {
                if (prevTemp == nullptr) {
                    Nodes = nextTemp;
                } else {
                    prevTemp->sibling = nextTemp;
                }

                temp->parent = nextTemp;
                temp->sibling = nextTemp->child;
                nextTemp->child = temp;
                nextTemp->degree++;
                temp = nextTemp;
            }
        }

        nextTemp = temp->sibling;
    }
}

void binomial_heap::insert(int value) {
    if (value > 0) {
        binomial_heap_node* temp = env->nodeAlloc();
        temp->setKey(value);
        if (Nodes == nullptr) {
            Nodes = temp;
            size = 1;
        } else {
            unionNodes(temp);
            size++;
        }
    }
}

int binomial_heap::extractMin() {
    if (Nodes == nullptr) {
        return -1;
    }

    binomial_heap_node* temp = Nodes, *prevTemp = nullptr;
    binomial_heap_node* minNode = Nodes->findMinNode();
    while (temp->key != minNode->key) {
        prevTemp = temp;
        temp = temp->sibling;
    }
    if (prevTemp == nullptr) {
        Nodes = temp->sibling;
    } else {
        prevTemp->sibling = temp->sibling;
    }

    temp = temp->child;
    binomial_heap_node* fakeNode = temp;
    while (temp != nullptr) {
        temp->parent = nullptr;
        temp = temp->sibling;
    }
    if ((Nodes == nullptr) && (fakeNode == nullptr)) {
        size = 0;
    } else {
        if ((Nodes == nullptr) && (fakeNode != nullptr)) {
            Nodes = fakeNode->reverse(nullptr);
            size = Nodes->getSize();
        } else {
            if ((Nodes != nullptr) && (fakeNode == nullptr)) {
                size = Nodes->getSize();
            } else {
                unionNodes(fakeNode->reverse(nullptr));
                size = Nodes->getSize();
            }
        }
    }
    int minKey = minNode->key;
    minNode->parent = nullptr;
    minNode->child = nullptr;
    minNode->sibling = nullptr;
    // delete minNode;

    return minKey;
}

void binomial_heap::decreaseKeyValue(int old_value, int new_value) {
    binomial_heap_node* temp = Nodes->findANodeWithKey(old_value);
    if (temp == nullptr) {
        return;
    }

    temp->key = new_value;
    binomial_heap_node* tempParent = temp->parent;
    while ((tempParent != nullptr) && (temp->key < tempParent->key)) {
        int z = temp->key;
        temp->key = tempParent->key;
        tempParent->key = z;
        temp = tempParent;
        tempParent = tempParent->parent;
    }
}

void binomial_heap::deleteNode(int value) {
    if ((Nodes != nullptr) && (Nodes->findANodeWithKey(value) != nullptr)) {
        decreaseKeyValue(value, findMinimum() - 1);
        extractMin();
    }
}
