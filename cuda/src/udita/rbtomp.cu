#include "rbt.h"
#include "../ompmains.h"

Node* nodeAlloc(Env *env) {
    // Node *node = (Node*) malloc(sizeof(Node));
    Node *node = &(env->pool[env->pix]);
    (env->pix)++;
    if (env->pix >= POOL_SIZE) {
        printf("ERROR: not enough objects in the pool\n");
    }

    node->key = 0;
    node->parent = NULL;
    node->left = NULL;
    node->right = NULL;
    return node;
}

void nodeUpdate(Env *env, Node *node, Node *parent, int32_t min, int32_t max) {
    node->parent = parent;
    node->key = _choice(min, max);
    node->color = _choice(RED, BLACK);
}

void initSubTree(Env *env, Node *node, int32_t size, int32_t min, int32_t max) {
    int32_t left_size = node->key - min;
    int32_t right_size = size - left_size;
    
    if (left_size == 0) {
        node->left = NULL;
    } else {
        node->left = nodeAlloc(env);
        nodeUpdate(env, node->left, node, min, node->key - 1);
        if (node->left->key == INVALID_VALUE || node->left->color == INVALID_VALUE) {
            return;
        }
        initSubTree(env, node->left, left_size - 1, min, node->key - 1);        
    }

    if (right_size == 0) {
        node->right = NULL;
    } else {
        node->right = nodeAlloc(env);
        nodeUpdate(env, node->right, node, node->key + 1, max);
        if (node->right->key == INVALID_VALUE || node->right->color == INVALID_VALUE) {
            return;
        }
        initSubTree(env, node->right, right_size - 1, node->key + 1, max);
    }      
}

int8_t repOkColorsRec(Env *env, Node *node) {
    if (node == NULL) {
        return 0;
    }

    // RedHasOnlyBlackChildren
    if (node->color == RED) {
        if (node->left != NULL && node->left->color == RED) {
            return ENV_FALSE;
        }
        if (node->right != NULL && node->right->color == RED) {
            return ENV_FALSE;
        }
    }

    // SimplePathsFromRootToNILHaveSameNumberOfBlackNodes
    int left_number = repOkColorsRec(env, node->left);
    if (left_number == ENV_FALSE) {
        return ENV_FALSE;
    }

    int right_number = repOkColorsRec(env, node->right);
    if (right_number == ENV_FALSE) {
        return ENV_FALSE;
    }
    if (left_number != right_number) {
        return ENV_FALSE;
    }

    return (left_number + (node->color == BLACK ? 1 : 0));
}

int8_t repOkColors(Env *env, RedBlackTree *rbt) {
    return repOkColorsRec(env, rbt->root) != ENV_FALSE;
}

void rbtGenerate(Env *env, RedBlackTree *rbt) {
    rbt->size = env->tree_size;
    rbt->root = nodeAlloc(env);
    nodeUpdate(env, rbt->root, NULL, 0, env->max_node_value);
    initSubTree(env, rbt->root, rbt->size - 1, 0, env->max_node_value);

    int8_t rep_ok = repOkColors(env, rbt);
    _countIf(rep_ok);
    _ignoreIf(!rep_ok);
}

void rbtUdita(int32_t size) {
    int tid = omp_get_thread_num();

    Node pool[POOL_SIZE];
    Env env = {
        .tree_size = size,
        .max_node_value = size - 1,
        .pix = 0,
        .pool = pool,
    };
    RedBlackTree rbt = {
        .size = 0,
        .root = NULL,
    };

    rbtGenerate(&env, &rbt);
}

int main(int argc, char *argv[]) {
    return uditaMainOMP(argc, argv, (void (*)(...))rbtUdita);
}