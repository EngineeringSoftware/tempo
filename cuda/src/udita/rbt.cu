#include "rbt.h"
#include "../mains.h"

__device__ Node* nodeAlloc(Env * const env);
__device__ void rbtPut(Env *env, RedBlackTree *rbt, int32_t key);
__device__ void rbtTreeInsert(Env *env, RedBlackTree *rbt, Node *z);
__device__ void rbtLeftRotate(RedBlackTree *rbt, Node *x);
__device__ void rbtRightRotate(RedBlackTree *rbt, Node *x);
__device__ void rbtTreeInsertFix(Env *env, RedBlackTree *rbt, Node *z);
__device__ int8_t getColor(Env *env, Node *x);
__device__ void setColor(Node *x, int8_t color);
__device__ int8_t repOk(Env *env, RedBlackTree *rbt);
__device__ int8_t orderedKeys(Env *env, Node *e, int min, int max);
__device__ int8_t repOkKeysAndValues(Env *env, RedBlackTree *rbt);
__device__ Node* parent(Node *n);
__device__ Node* left(Node *n);
__device__ Node* right(Node *n);

__host__ __device__ void llInit(LinkedList *ll) {
    ll->size = 0;
    ll->first_index = 0;
    ll->last_index = 0;
}

__host__ __device__ void llAdd(LinkedList *ll, Node *n) {
    if (ll->last_index + 1 >= MAX_LL_SIZE) {
        printf("ERROR: max linked list size limit reached\n");
        // asm("exit;");
    }

    ll->nodes[ll->last_index].value = n;
    ll->last_index++;
    ll->size++;
}

__host__ __device__ int8_t llRemove(LinkedList *ll, Node *n) {
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

__host__ __device__ Node* llRemoveFirst(LinkedList *ll) {
    if (ll->size == 0) {
        return NULL;
    }

    Node *result = ll->nodes[ll->first_index++].value;
    ll->size--;

    return result;
}

__host__ __device__ Node* llRemoveLast(LinkedList *ll) {
    assert(ll->size > 0);

    ll->last_index--;
    Node *result = ll->nodes[ll->last_index].value;
    ll->size--;

    return result;
}

__host__ __device__ int8_t llContains(LinkedList *ll, Node *n) {
    assert(ll != NULL);
    for (int32_t i = ll->first_index; i < ll->last_index; i++) {
        if (ll->nodes[i].value == n) {
            return TRUE;
        }
    }
    return FALSE;
}

/* ---------------------------------------- */

__host__ __device__ int32_t setSize(Set *s) {
    return s->ll->size;
}

__host__ __device__ void setInit(Set *s, LinkedList *ll) {
    llInit(ll);
    s->ll = ll;
}

__host__ __device__ void setAdd(Set *s, Node *n) {
    assert(s != NULL);
    assert(s->ll != NULL);

    if (!setContains(s, n)) {
        llAdd(s->ll, n);
    }
}

__host__ __device__ void setRemove(Set *s, Node *n) {
    llRemove(s->ll, n);
}

__host__ __device__ int8_t setContains(Set *s, Node *n) {
    assert(s != NULL);
    assert(s->ll != NULL);
    return llContains(s->ll, n);
}


// Needed for running a test

__device__ void rbtPut(Env *env, RedBlackTree *rbt, int32_t key) {
    Node* x = nodeAlloc(env);
    x->key = key;
    rbtTreeInsert(env, rbt, x);
    rbt->size++;
}

__device__ void rbtTreeInsert(Env *env, RedBlackTree *rbt, Node *z) {
    Node* y = NULL;
    Node* x = rbt->root;
    while (x != NULL) {
        y = x;
        if (z->key < x->key) {
            x = x->left;
        } else {
            if (z->key == x->key) {
                rbt->size--;
                return;
            } else {
                x = x->right;
            }
        }
    }
    z->parent = y;
    if (y == NULL) {
        rbt->root = z;
    } else {
        if (z->key < y->key) {
            y->left = z;
        } else {
            y->right = z;
        }
    }

    z->left = NULL;
    z->right = NULL;
    z->color = RED;
    rbtTreeInsertFix(env, rbt, z);
}

__device__ int8_t getColor(Env *env,Node *x) {
    return x == NULL ? BLACK : x->color;
}

__device__ void setColor(Node *x, int8_t color) {
    if (x != NULL) {
        x->color = color;
    }
}

__device__ Node* parent(Node *n) {
    return n == NULL ? NULL : n->parent;
}

__device__ Node* left(Node *n) {
    return n == NULL ? NULL : n->left;
}

__device__ Node* right(Node *n) {
    return n == NULL ? NULL : n->right;
}

__device__ void rbtLeftRotate(RedBlackTree *rbt,Node *x) {
    Node* y = x->right;
    x->right = y->left;
    if (y->left != NULL) {
        y->left->parent = x;
    }

    y->parent = x->parent;
    if (x->parent == NULL) {
        rbt->root = y;
    } else {
        if (x == x->parent->left) {
            x->parent->left = y;
        } else {
            x->parent->right = y;
        }
    }

    y->left = x;
    x->parent = y;
}

__device__ void rbtRightRotate(RedBlackTree *rbt, Node *x) {
    Node* y = x->left;
    x->left = y->right;
    if (y->right != NULL) {
        y->right->parent = x;
    }

    y->parent = x->parent;
    if (x->parent == NULL) {
        rbt->root = y;
    } else {
        if (x == x->parent->right) {
            x->parent->right = y;
        } else {
            x->parent->left = y;
        }
    }

    y->right = x;
    x->parent = y;
}

__device__ void rbtTreeInsertFix(Env *env, RedBlackTree *rbt, Node *z) {
    while (getColor(env, z->parent) == RED) {
        if (parent(z) == left(parent(parent(z)))) {
            Node* y = right(parent(parent(z)));
            if (getColor(env, y) == RED) {
                setColor(parent(z), BLACK);
                setColor(y, BLACK);
                setColor(parent(parent(z)), RED);
                z = parent(parent(z));
            } else {
                if (z == right(parent(z))) {
                    z = parent(z);
                    rbtLeftRotate(rbt, z);
                }

                setColor(parent(z), BLACK);
                setColor(parent(parent(z)), RED);
                if (parent(parent(z)) != NULL) {
                    rbtRightRotate(rbt, parent(parent(z)));
                }
            }
        } else {
            Node* y = left(parent(parent(z)));
            if (getColor(env, y) == RED) {
                setColor(parent(z), BLACK);
                setColor(y, BLACK);
                setColor(parent(parent(z)), RED);
                z = parent(parent(z));
            } else {
                if (z == left(parent(z))) {
                    z = parent(z);
                    rbtRightRotate(rbt, z);
                }

                setColor(parent(z), BLACK);
                setColor(parent(parent(z)), RED);
                if (parent(parent(z)) != NULL) {
                    rbtLeftRotate(rbt, parent(parent(z)));
                }
            }
        }
    }

    rbt->root->color = BLACK;
}

// ----------------------------------------

__device__ Node* nodeAlloc(Env *const env) {
    Node *const node = &(env->pool[env->pix]);
    (env->pix)++;

    #ifndef DISABLE_CHOICE_DEBUG
    if (env->pix >= RBT_POOL_SIZE) {
        printf("ERROR: not enough objects in the pool\n");
    }
    #endif

    node->key = 0;
    node->parent = NULL;
    node->left = NULL;
    node->right = NULL;
    return node;
}

// LOC.py start

__forceinline__ __device__ void nodeUpdate(Node *const node, Node *const parent, const int8_t min, const int8_t max, const int8_t size) {
    node->min = min;
    node->max = max;
    node->size = size;
    node->parent = parent;
    int x = (min == max) ? _choice(RED, BLACK) : _choice(min, min + ((max - min) << 1) + 1);
    node->key = (min == max) ? min : min + ((x - min) >> 1);
    node->color = x % 2;
}

__device__ void initSubTree(Env *const env) {
    for (int i = 0; i < env->max_node_value; i++) {
        Node *const node = &(env->pool[i]);

        const int8_t min = node->min;
        const int8_t max = node->max;
        const int8_t size = node->size;
        const int8_t left_size = node->key - min;
        const int8_t right_size = size - left_size;

        if (left_size != 0) {
            node->left = nodeAlloc(env);
            nodeUpdate(node->left, node, min, node->key - 1, left_size - 1);
        } else {
            node->left = NULL;
        }

        if (right_size != 0) {
            node->right = nodeAlloc(env);
            nodeUpdate(node->right, node, node->key + 1, max, right_size -1 );
        } else {
            node->right = NULL;
        }
    }
}

__device__ int8_t repOkColorsRec(Env *const env, Node *const node) {
    if (node == NULL)
        return 0;

    // RedHasOnlyBlackChildren
    if (node->color == RED) {
        if (node->left != NULL && node->left->color == RED)
            return ENV_FALSE;
        if (node->right != NULL && node->right->color == RED)
            return ENV_FALSE;
    }

    // SimplePathsFromRootToNILHaveSameNumberOfBlackNodes
    const int left_number = repOkColorsRec(env, node->left);
    if (left_number == ENV_FALSE)
        return ENV_FALSE;

    const int right_number = repOkColorsRec(env, node->right);
    if (right_number == ENV_FALSE)
        return ENV_FALSE;
    if (left_number != right_number)
        return ENV_FALSE;

    /* originally (left_number + (node->color == BLACK ? 1 : 0)), but BLACK == 1 and RED == 0 */
    return (left_number + node->color);
}

__device__ int8_t repOkColors(Env *const env, RedBlackTree *const rbt) {
    return repOkColorsRec(env, rbt->root) != ENV_FALSE;
}


// LOC.py ignore start

__device__ int8_t repOkKeysAndValues(Env *const env, RedBlackTree *const rbt) {
    if (!orderedKeys(env, rbt->root, -1, -1)) {
        return 0;
    }

    LinkedList work;
    llInit(&work);

    llAdd(&work, rbt->root);
    while (work.size != 0) {
        Node *current = llRemoveFirst(&work);

        if (current->left != NULL) {
            llAdd(&work, current->left);
        }
        if (current->right != NULL) {
            llAdd(&work, current->right);
        }
    }

    return TRUE;
}

// use int instead of Java Object for min and max, with -1 instead of NULL
__device__ int8_t orderedKeys(Env *const env, Node *const e, const int min, const int max) {
    if (e->key == -1) {
        return FALSE;
    }

    if (((min != -1) && e->key <= min)
        || ((max != -1) && e->key >= max)) {
        return FALSE;
    }

    if (e->left != NULL) {
        if (!orderedKeys(env, e->left, min, e->key)) {
            return FALSE;
        }
    }

    if (e->right != NULL) {
        if (!orderedKeys(env, e->right, e->key, max)) {
            return FALSE;
        }
    }

    return TRUE;
}

__device__ int8_t repOk(Env *const env, RedBlackTree *const rbt) {
    if (rbt->root == NULL) {
        return rbt->size == 0;
    }

    // RootHasNoParent
    if (rbt->root->parent != NULL) {
        return FALSE;
    }

    LinkedList ll_visited;
    Set visited;
    setInit(&visited, &ll_visited);
    setAdd(&visited, rbt->root);

    LinkedList work;
    llInit(&work);

    llAdd(&work, rbt->root);
    while (work.size != 0) {
        Node *current = llRemoveFirst(&work);

        Node *cl = current->left;
        if (cl != NULL) {
            if (setContains(&visited, cl)) {
                return FALSE;
            }
            setAdd(&visited, cl);

            if (cl->parent != current) {
                return FALSE;
            }

            llAdd(&work, cl);
        }

        Node *cr = current->right;
        if (cr != NULL) {
            if (setContains(&visited, cr)) {
                return FALSE;
            }
            setAdd(&visited, cr);

            if (cr->parent != current) {
                return FALSE;
            }

            llAdd(&work, cr);
        }
    }

    // SizeOk
    if (setSize(&visited) != rbt->size) {
        return FALSE;
    }

    if (repOkColorsRec(env, rbt->root) == ENV_FALSE)
        return FALSE;

    return repOkKeysAndValues(env, rbt);
}

// LOC.py ignore stop

__device__ void rbtGenerate(Env *const env, RedBlackTree *const rbt) {
    rbt->size = env->tree_size;
    rbt->root = nodeAlloc(env);
    nodeUpdate(rbt->root, NULL, 0, env->max_node_value, env->max_node_value);
    initSubTree(env);

    const int8_t rep_ok = repOkColors(env, rbt);
    _countIf(rep_ok);
    _ignoreIf(!rep_ok);
}

__global__ void rbtUdita(const int32_t bck_active, const int32_t size) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    Node pool[RBT_POOL_SIZE];
    Env env = { .tree_size = size, .max_node_value = size - 1, .pix = 0, .pool = pool,};
    RedBlackTree rbt = { .size = 0, .root = NULL,};

    rbtGenerate(&env, &rbt);
// LOC.py stop

    /* This is the alternative repOk used in korat */
    // int8_t rep_ok = repOk(&env, &rbt);
    // _countIf(rep_ok);
    // _ignoreIf(!rep_ok);

    #ifdef RUN_TEST
    rbtPut(&env, &rbt, (int32_t)idx);
    #endif
}


int main(int argc, char *argv[]) {
    size_t limit;
    CUDA_CHECK_RETURN(cudaDeviceGetLimit(&limit, cudaLimitStackSize));
    CUDA_CHECK_RETURN(cudaDeviceSetLimit(cudaLimitStackSize, limit * 10));
    return uditaMain(argc, argv, (void (*)(...)) rbtUdita);
}