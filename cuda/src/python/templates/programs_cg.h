
#include <stdint.h>
#define MAX_CHILDREN 10

typedef struct _callgraphnode {
    int32_t num_of_children;
    int8_t function_id;
    struct _callgraphnode *children[MAX_CHILDREN];
} Node;

typedef struct _callgraph {
    Node *root;
} CG;