
#include <assert.h>
#include "dag.h"

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
