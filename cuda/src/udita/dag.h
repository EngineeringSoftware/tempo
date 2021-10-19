/*
 * This file is implementation of
 * assume.dag.DAG from the UDITA
 * repository.
 */

#ifndef DAG_H
#define DAG_H

#include <stdint.h>
#include "../objpool.h"

/* max children per node */
#define MAX_CHILDREN 20

typedef struct _dagnode {
    int32_t num_of_children;
    int8_t id;
    struct _dagnode *children[MAX_CHILDREN];
} Node;

DefObjPool(Node);

/* ---------------------------------------- */

typedef struct _dag_env {
    /* intended size of the graph */
    int32_t num_of_nodes;
    /* object pool */
    NodePool *op;
} Env;

/* ---------------------------------------- */

typedef struct _dag {
    Node *root;
} DAG;

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

__host__ __device__ void llInit(LinkedList *ll);
__host__ __device__ void llAdd(LinkedList *ll, Node *n);
__host__ __device__ int8_t llRemove(LinkedList *ll, Node *n);
__host__ __device__ Node* llRemoveFirst(LinkedList *ll);
__host__ __device__ Node* llRemoveLast(LinkedList *ll);
__host__ __device__ int8_t llContains(LinkedList *ll, Node *n);

/* ---------------------------------------- */

typedef struct _set {
    LinkedList *ll;
} Set;

__host__ __device__ void setInit(Set *s, LinkedList *ll);
__host__ __device__ int32_t setSize(Set *s);
__host__ __device__ void setAdd(Set *s, Node *n);
__host__ __device__ void setRemove(Set *s, Node *n);
__host__ __device__ int8_t setContains(Set *s, Node *n);

#endif
