
#include "st.h"
#include "../ompmains.h"
#include "../consts.h"

Node* nodeAlloc(Env *env) {
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

void nodeUpdate(Env *env, Node *node) {
    node->value = _choice(0, env->max_node_value);
}

void initSubTree(Env *env, Node *node, int32_t size) {
    int left_size = _choice(0, size);
    int right_size = size - left_size;

    if (left_size != 0 && left_size != INVALID_VALUE) {
        node->left = nodeAlloc(env);
        nodeUpdate(env, node->left);
        initSubTree(env, node->left, left_size - 1);
    }

    if (right_size != 0 && left_size != INVALID_VALUE) {
        node->right = nodeAlloc(env);
        nodeUpdate(env, node->right);
        initSubTree(env, node->right, right_size - 1);
    }
}

int8_t repOkIsOrdered(Node *node, int min, int max) {
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

int8_t orderProperty(SearchTree *st) {
    return repOkIsOrdered(st->root, -1, -1);
}

void stGenerate(Env *env, SearchTree *st) {
    st->size = env->tree_size;
    st->root = nodeAlloc(env);
    nodeUpdate(env, st->root);
    initSubTree(env, st->root, st->size - 1);

    int8_t is_ordered = orderProperty(st);
    _countIf(is_ordered);
    _ignoreIf(!is_ordered);
}

void stUdita(int32_t size) {
    int tid = omp_get_thread_num();

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
}

int main(int argc, char *argv[]) {
    return uditaMainOMP(argc, argv, (void (*)(...))stUdita);
}
