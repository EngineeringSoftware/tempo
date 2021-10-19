
/*
 * This file is implementation of
 * oracle.searchtree.SearchTree
 * from the UDITA repository.
 */

#ifndef _ST_H
#define _ST_H

#include <stdint.h>

typedef struct _searchtreenode {
    int32_t value;
    struct _searchtreenode *left;
    struct _searchtreenode *right;
} Node;

typedef struct _searchtree {
    int32_t size;
    Node *root;
} SearchTree;

typedef struct _searchtree_env {
    int32_t tree_size;
    int32_t max_node_value;
    int8_t pix;
    Node *pool;
} Env;

// ----------------------------------------

#endif
