
/*
 * This file is implementation of
 * oracle.redblacktree.RedBlackTree
 * from the UDITA repository.
 */

#ifndef _RBTUDITA_H
#define _RBTUDITA_H

#include <stdint.h>
#include "../consts.h"

#define RED 0
#define BLACK 1
#define ENV_FALSE -1

typedef struct _redblacktreenode {
    int8_t color;
    int8_t key;
    int8_t min;
    int8_t max;
    int8_t size;
    struct _redblacktreenode *parent;
    struct _redblacktreenode *left;
    struct _redblacktreenode *right;
} Node;

typedef struct _redblacktree {
    int32_t size;
    Node *root;
} RedBlackTree;

typedef struct _redblacktree_env {
    int32_t tree_size;
    int32_t max_node_value;
    int8_t pix;
    Node *pool;
} Env;

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