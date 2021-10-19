#ifndef IGCOMPLEX_H
#define IGCOMPLEX_H

#include <stdint.h>
#include "../objpool.h"

/* max children per node */
#define MAX_CHILDREN 20
#define MAX_IGCOMPLEX_EDGES 2;
#define MAX_BOOL_CHOICE 1;
#define MAX_METHOD_TYPE 2;

typedef struct _ignode {
    int32_t num_of_children;
     // 0 - no , 1 - yes
    int8_t has_method;
    // 0 - pure virtual, 1 - virtual no body, 2 - normal body
    int8_t method_type;
     // 0 - no , 1 - yes
    int8_t has_method_arg;
    // 0 - public, 1 - protected, 2 - private, 3 - virtual public
    int8_t inheritance_type;
    int8_t is_virtual;
    int8_t id;
    // 0 - unvisited, 1 - visiting, 2 - visited
    int8_t visited;
    struct _ignode *children[MAX_CHILDREN];
} Node;

DefObjPool(Node);

/* ---------------------------------------- */

typedef struct _ig_env {
    /* intended size of the graph */
    int32_t num_of_nodes;
    /* object pool */
    NodePool *op;
} Env;

/* ---------------------------------------- */

typedef struct _ig {
    Node *root;
} IG;

/*
 * Generates IGs for the given environment.
 */
__device__ void igGenerate(Env*, IG*);
__device__ void igPrint(IG*, char []);

/* ---------------------------------------- */

#define MAX_LL_SIZE 20

typedef struct _linkedlistnode {
    Node *value;
} LLNode;

typedef struct _linkedlist {
    int32_t size;
    int32_t first_index;
    int32_t last_index;
    LLNode nodes[MAX_LL_SIZE];
} LinkedList;

__device__ void llInit(LinkedList *ll);
__device__ void llAdd(LinkedList *ll, Node *n);
__device__ int8_t llRemove(LinkedList *ll, Node *n);
__device__ Node* llRemoveFirst(LinkedList *ll);
__device__ Node* llRemoveLast(LinkedList *ll);
__device__ int8_t llContains(LinkedList *ll, Node *n);

/* ---------------------------------------- */

typedef struct _set {
    LinkedList *ll;
} Set;

__device__ void setInit(Set *s, LinkedList *ll);
__device__ int32_t setSize(Set *s);
__device__ void setAdd(Set *s, Node *n);
__device__ void setRemove(Set *s, Node *n);
__device__ int8_t setContains(Set *s, Node *n);

__device__ void llInit(LinkedList *ll) {
    ll->size = 0;
    ll->first_index = 0;
    ll->last_index = 0;
}

__device__ void llAdd(LinkedList *ll, Node *n) {
    if (ll->last_index + 1 >= MAX_LL_SIZE) {
        printf("ERROR: max linked list size limit reached\n");
        asm("exit;");
    }

    ll->nodes[ll->last_index].value = n;
    ll->last_index++;
    ll->size++;
}

__device__ int8_t llRemove(LinkedList *ll, Node *n) {
    if (ll->size == 0) {
        return FALSE;
    }

    for (int32_t i = ll->first_index; i < ll->last_index; i++) {
        if (ll->nodes[i].value == n) {
            for (int32_t j = i; j < ll->last_index; j++) {
                ll->nodes[j].value = ll->nodes[j + 1].value;
            }
            ll->last_index--;
            ll->size--;

            return TRUE;
        }
    }

    return FALSE;
}

__device__ Node* llRemoveFirst(LinkedList *ll) {
    if (ll->size == 0) {
        return NULL;
    }

    Node *result = ll->nodes[ll->first_index++].value;
    ll->size--;

    return result;
}

__device__ Node* llRemoveLast(LinkedList *ll) {
    assert(ll->size > 0);

    ll->last_index--;
    Node *result = ll->nodes[ll->last_index].value;
    ll->size--;

    return result;
}

__device__ int8_t llContains(LinkedList *ll, Node *n) {
    assert(ll != NULL);
    for (int32_t i = ll->first_index; i < ll->last_index; i++) {
        if (ll->nodes[i].value == n) {
            return TRUE;
        }
    }
    return FALSE;
}

/* ---------------------------------------- */

__device__ int32_t setSize(Set *s) {
    return s->ll->size;
}

__device__ void setInit(Set *s, LinkedList *ll) {
    llInit(ll);
    s->ll = ll;
}

__device__ void setAdd(Set *s, Node *n) {
    assert(s != NULL);
    assert(s->ll != NULL);

    if (!setContains(s, n)) {
        llAdd(s->ll, n);
    }
}

__device__ void setRemove(Set *s, Node *n) {
    llRemove(s->ll, n);
}

__device__ int8_t setContains(Set *s, Node *n) {
    assert(s != NULL);
    assert(s->ll != NULL);
    return llContains(s->ll, n);
}

#endif
