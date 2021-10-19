
#include "st.h"
#include "../mains.h"

__device__ Node* nodeAlloc(Env *env);

// Needed for running a test

__device__ void stInsert(Env *env, SearchTree *const st, const int32_t value) {
    st->size++;
    Node *z = nodeAlloc(env);
    z->value = value;

    Node *y = NULL;
    Node *x = st->root;

    while (x != NULL) {
        y = x;
        if (z->value < x->value) {
            x = x->left;
        } else {
            if (z->value == x->value) {
                st->size--;
                return;
            } else {
                x = x->right;
            }
        }
    }

    if (y == NULL) {
        st->root = z;
    } else {
        if (z->value < y->value) {
            y->left = z;
        } else {
            y->right = z;
        }
    }
}

// ----------------------------------------


// LOC.py start
__device__ Node* nodeAlloc(Env *env) {
    // Node *node = (Node*) malloc(sizeof(Node));
    Node *node = &(env->pool[env->pix]);
    (env->pix)++;
    if (env->pix >= POOL_SIZE) {
        printf("ERROR: not enough objects in the pool\n");
    }

    node->value = 0;
    node->left = NULL;
    node->right = NULL;
    return node;
}

__device__ void nodeUpdate(Env *env, Node *const node) {
    node->value = _choice(0, env->max_node_value);
}

__device__ void initSubTree(Env *env, Node *const node, const int32_t size) {
    int left_size = _choice(0, size);
    int right_size = size - left_size;

    if (left_size != 0) {
        node->left = nodeAlloc(env);
        nodeUpdate(env, node->left);
        initSubTree(env, node->left, left_size - 1);
    }

    if (right_size != 0) {
        node->right = nodeAlloc(env);
        nodeUpdate(env, node->right);
        initSubTree(env, node->right, right_size - 1);
    }
}

__device__ int8_t repOkIsOrdered(Node *const node, const int min, const int max) {
    if (node->value == -1) {
        return 0;
    }
    if ((min != -1 && node->value < (min)) || (max != -1 && node->value > (max))) {
        return 0;
    }
    if (node->left != NULL) {
        if (!repOkIsOrdered(node->left, min, node->value)) {
            return 0;
        }
    }
    if (node->right != NULL) {
        if (!repOkIsOrdered(node->right, node->value, max)) {
            return 0;
        }
    }

    return 1;
}

__device__ int8_t orderProperty(SearchTree *const st) {
    return repOkIsOrdered(st->root, -1, -1);
}

__device__ void stGenerate(Env *const env, SearchTree *const st) {
    st->size = env->tree_size;
    st->root = nodeAlloc(env);
    nodeUpdate(env, st->root);
    initSubTree(env, st->root, st->size - 1);

    int8_t is_ordered = orderProperty(st);
    _countIf(is_ordered);
    _ignoreIf(!is_ordered);
}

__global__ void stUdita(const int32_t bck_active, const int32_t size) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    Node pool[POOL_SIZE];
    Env env = {
        .tree_size = size,
        .max_node_value = size - 1,
        .pix = 0,
        .pool = pool,
    };
    SearchTree st = {
        .size = 0,
        .root = NULL,
    };

    stGenerate(&env, &st);
    #ifdef RUN_TEST
    stInsert(&env, &st, (int32_t)idx);
    #endif
}
// LOC.py stop


int main(int argc, char *argv[]) {
    return uditaMain(argc, argv, (void (*)(...)) stUdita);
}
