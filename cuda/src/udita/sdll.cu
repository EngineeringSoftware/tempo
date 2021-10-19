
#include <stdio.h>
#include <stdlib.h>
#include "sdll.h"
#include "../mains.h"
#include "../consts.h"

__device__ SDLLNode* sdllNodeAlloc(Env *env);

// Needed for running a test

__device__ void sdllInsert(Env *const env, SDLL *const sdll, const int32_t value) {
    SDLLNode *node = sdllNodeAlloc(env);
    if (sdll->head == NULL) {
        sdll->head = node;
    } else {
        SDLLNode *curr = sdll->head;
        SDLLNode *bef = NULL;

        while (curr != NULL) {
            // find first element which has bigger value
            if (DACCESS(curr->value) > DACCESS(node->value)) {
                node->prev = bef;
                node->next = curr;
                curr->prev = node;
                if (bef == NULL) {
                    // first elem
                    sdll->head = node;
                } else {
                    bef->next = node;
                }
                break;
            }
            bef = curr;
            curr = curr->next;
        }
        if (curr == NULL) {
            bef->next = node;
            node->prev = bef;
        }
    }
    sdll->size++;
}

// ----------------------------------------

// LOC.py start
__device__ SDLLNode* sdllNodeAlloc(Env *const env) {
    SDLLNode *node = &(env->pool[env->pix]);
    (env->pix)++;
    if (env->pix >= POOL_SIZE) {
        printf("ERROR: not enough objects in the pool\n");
    }

    node->value = { 0 };
    node->next = 0;
    node->prev = 0;
    return node;
}

__device__ void sdllNodeUpdate(const Env *const env, SDLLNode *const node, const int32_t min_value) {
    DCHOICE(node->value, min_value, env->max_value);
}

__device__ void sdllGenerateUnsorted(Env *const env, SDLL *const sdll) {
    sdll->size = _choice(env->min_size, env->max_size);

    SDLLNode *current = NULL;
    for (int32_t i = 0; i < sdll->size; i++) {
        if (current == NULL) {
            sdll->head = sdllNodeAlloc(env);
            sdllNodeUpdate(env, sdll->head, env->min_value);
            current = sdll->head;
        } else {
            current->next = sdllNodeAlloc(env);
            sdllNodeUpdate(env, current->next, DACCESS(current->value));
            current->next->prev = current;
            current = current->next;
        }
    }
}

__device__ int8_t sdllIsSorted(SDLL *const sdll) {
    if (sdll->size == 1) {
        // original code has the following line to force delayed execution
        const int32_t value = DACCESS(sdll->head->value);
    } else if (sdll->size > 0) {
        SDLLNode *tmp1 = sdll->head;
        SDLLNode *tmp2 = sdll->head->next;
        while (tmp2 != NULL) {
            if (DACCESS(tmp1->value) > DACCESS(tmp2->value)) {
                return 0;
            }
            tmp1 = tmp2;
            tmp2 = tmp1->next;
        }
    }
    return 1;
}

__device__ void sdllGenerate(Env *const env, SDLL *const sdll) {
    sdllGenerateUnsorted(env, sdll);
    const int8_t is_sorted = sdllIsSorted(sdll);
    // original code has the following line to count structures that
    // satisfy the property
    _countIf(is_sorted);
    _ignoreIf(!is_sorted);
}

__global__ void sdllUdita(const int32_t bck_active, const int32_t size) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    SDLLNode pool[POOL_SIZE] = { 0 };
    Env env = {
        .min_value = 0,
        .max_value = size - 1,
        .min_size = 0,
        .max_size = size,
        .pix = 0,
        .pool = pool,
    };
    SDLL sdll = {
        .head = NULL,
        .size = 0,
    };
    
    sdllGenerate(&env, &sdll);
    #ifdef RUN_TEST
    sdllInsert(&env, &sdll, (int32_t)idx);
    #endif

}
// LOC.py stop


int main(int argc, char *argv[]) {
    return uditaMain(argc, argv, (void (*)(...)) sdllUdita);
}
