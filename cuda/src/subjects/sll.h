
/*
 * This is C++ version of
 * predicate.taco.singlylist.SinglyLinkedList_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef SLL_H_
#define SLL_H_

#include <stdio.h>
#include "../consts.h"

class Object;
struct SLLNode;

class Env {
public:
    Object *objectPool;
    SLLNode *nodePool;
    int8_t objectPoolIndex;
    int8_t nodePoolIndex;

    __host__ __device__ Env(Object *objectPool, SLLNode *nodePool) : objectPool(objectPool), nodePool(nodePool), objectPoolIndex(0), nodePoolIndex(0) {}

    __host__ __device__ SLLNode *nodeAlloc();
    __host__ __device__ Object *objectAlloc();
};

/* Fake class to get closer to Java implementation */
class Object {
 private:
    int id;

 public:
    __device__ __host__ Object(): id(0) {}
    __device__ __host__ Object(int val): id(val) {}
    __device__ __host__ void print() { printf("%d", id); }
};

struct SLLNode {
    SLLNode *next;
    Object *value; // Object or template?

    __device__ __host__ SLLNode(): next(nullptr), value(nullptr) {}

    __device__ __host__ SLLNode(const SLLNode &other) {
        value = other.value;
        if (other.next == nullptr) {
            next = nullptr;
        } else {
            next = new SLLNode(*other.next);
        }
    }

    __device__ __host__ void print() {
        if (value == nullptr) {
            printf("null");
        } else {
            value->print();
        }
    }
};

class SLL {
 private:
    SLLNode *header;
    Env *env;

    __device__ __host__ void copy(const SLL &other) {
        if (other.header == nullptr) {
            header = nullptr;
        } else {
            header = new SLLNode(*other.header);
        }
    }

    __device__ __host__ void destroy() {
        SLLNode *current = header;
        SLLNode *prev = nullptr;
        while (current != nullptr) {
            prev = current;
            current = current->next;
        }
    }

 public:

    __device__ __host__ SLL(Env *env): header(nullptr), env(env) {}

    __device__ __host__ ~SLL() { destroy(); }

    __device__ __host__ SLL(const SLL &other) {
        copy(other);
    }

    __device__ __host__ SLL& operator=(const SLL &other) {
        if (this == &other) {
            return *this;
        }
        destroy();
        copy(other);
        return *this;
    }

    __device__ __host__ bool contains(Object *value_param) {
        SLLNode *current = header;
        bool result = false;

        while (result == false && current != nullptr) {
            bool equal_val;
            if (value_param == nullptr && current->value == nullptr) {
                equal_val = true;
            } else {
                if (value_param != nullptr) {
                    if (value_param == current->value) {
                        equal_val = true;
                    } else {
                        equal_val = false;
                    }
                } else {
                    equal_val = false;
                }
            }

            if (equal_val == true) {
                result = true;
            }
            current = current->next;
        }

        return result;
    }

    __device__ __host__ void remove(int index) {
        SLLNode *current = header;
        SLLNode *previous = nullptr;
        int current_index = 0;
        bool found = false;

        while (!found && current != nullptr) {
            if (index == current_index) {
                found = true;
            } else {
                current_index++;
                previous = current;
                current = current->next;
            }
        }

        if (!found) {
            return;
        }

        if (previous == nullptr) {
            header = current->next;
        } else {
            previous->next = current->next;
        }
    }

    __device__ __host__ void insertBack(Object *arg) {
        SLLNode *fresh_node = env->nodeAlloc();
        fresh_node->value = arg;
        fresh_node->next = nullptr;
        if (header == nullptr) {
            header = fresh_node;
        } else {
            SLLNode *current = header;
            while (current->next != nullptr) {
                current = current->next;
            }
            current->next = fresh_node;
        }
    }

    __device__ __host__ void print() {
        printf("sll: ");
        if (header == nullptr) {
            printf("null\n");
            return;
        }

        SLLNode *c = header;
        c->print();
        while (c->next != nullptr) {
            c = c->next;
            printf(":");
            c->print();
        }
        printf("\n");
    }
};

SLLNode *Env::nodeAlloc() {
    SLLNode *new_node = &(nodePool[nodePoolIndex]);
    nodePoolIndex++;
    if (nodePoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in nodePool, Index %d\n", nodePoolIndex);
    }

    return new_node;
}

Object *Env::objectAlloc() {
    Object *new_object = &(objectPool[objectPoolIndex]);
    objectPoolIndex++;
    if (objectPoolIndex >= POOL_SIZE) {
        printf("ERROR: not enough objects in objectPool, Index %d\n", objectPoolIndex);
    }

    return new_object;
}

#endif
