#ifndef CG_H
#define CG_H

#include <stdint.h>
#include "../objpool.h"

/* max children per node */
#define MAX_CHILDREN 20

typedef struct _callgraphnode {
    int32_t num_of_children;
    int8_t function_id;
    struct _callgraphnode *children[MAX_CHILDREN];
} Node;

DefObjPool(Node);

/* ---------------------------------------- */

typedef struct _callgraph_env {
    /* intended size of the graph */
    int32_t num_of_nodes;
    /* object pool */
    NodePool *op;
} Env;

/* ---------------------------------------- */

typedef struct _callgraph {
    Node *root;
} CG;

__device__ void cgGenerate(Env*, CG*);

#endif