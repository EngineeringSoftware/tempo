/*
 * Copyright (R) Ahmet Celik
 */

#include "bt.h"
#include <iostream>

#include "../consts.h"

Node* Env::nodeAlloc() {
    Node *new_node = &(nodes[nodesIndex]);
    nodesIndex++;
    if (nodesIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodes, Index %d\n", nodesIndex);
    }

    return new_node;
}

__device__ __host__ void BT::add(int x) {
    if (root == nullptr) {
        root = env->nodeAlloc();
        root->setValue(x);
        return;
    }
    Node *current = root;
    int currentValue = current->getValue();
    while (currentValue != x) {
        if (x < currentValue) {
            if (current->getLeft() == nullptr) {
                Node *temp = env->nodeAlloc();
                temp->setValue(x);
                current->setLeft(temp);
            } else {
                current = current->getLeft();
                currentValue = current->getValue();
            }
        } else {
            if (current->getRight() == nullptr) {
                Node *temp = env->nodeAlloc();
                temp->setValue(x);
                current->setRight(temp);
            } else {
                current = current->getRight();
                currentValue = current->getValue();
            }
        }
    }
}

__device__ __host__ bool BT::remove(int x) {
    if (root != nullptr && root->getValue() == x &&
        root->getLeft() == nullptr && root->getRight() == nullptr) {
        root = nullptr;
        return true;
    }
    Node *current = root;
    Node *parent = nullptr;
    bool branch = true;
    while (current != nullptr) {
        if (x == current->getValue()) {
            if (current->getLeft() == nullptr &&
                current->getRight() == nullptr) {
                if (parent == nullptr) {
                    root = nullptr;
                } else {
                    if (branch) {
                        parent->setLeft(nullptr);
                    } else {
                        parent->setRight(nullptr);
                    }
                }
            } else if (current->getLeft() == nullptr) {
                if (parent == nullptr) {
                    root = current->getRight();
                } else {
                    if (branch) {
                        parent->setLeft(current->getRight());
                    } else {
                        parent->setRight(current->getRight());
                    }
                }
            } else if (current->getRight() == nullptr) {
                if (parent == nullptr) {
                    root = current->getLeft();
                } else {
                    if (branch) {
                        parent->setLeft(current->getLeft());
                    } else {
                        parent->setRight(current->getLeft());
                    }
                }
            } else {
                parent = current;
                Node *bigson = current->getLeft();
                branch = true;
                while (bigson->getLeft() != nullptr || bigson->getRight() != nullptr) {
                    parent = bigson;
                    if (bigson->getRight() != nullptr) {
                        bigson = bigson->getRight();
                        branch = false;
                    } else {
                        bigson = bigson->getLeft();
                        branch = true;
                    }
                }
                if (parent != nullptr) {
                    if (branch) {
                        parent->setLeft(nullptr);
                    } else {
                        parent->setRight(nullptr);
                    }
                }
                if (bigson != current) {
                    current->setValue(bigson->getValue());
                }
            }
            return true;
        }
        parent = current;
        if (current->getValue() > x) {
            current = current->getLeft();
            branch = true;
        } else {
            current = current->getRight();
            branch = false;
        }
    }
    return false;
}

__device__ __host__ void Node::print() {
    if (left != nullptr) {
        left->print();
    }
    printf("%d ", value);
    if (right != nullptr) {
        right->print();
    }
}

__device__ __host__ void BT::print() {
    if (root != nullptr) {
        printf("bt: ");
        root->print();
    } else {
        printf("bt: null");
    }
    printf("\n");
}
