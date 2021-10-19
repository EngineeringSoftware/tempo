
#include <stdio.h>
#include <stdlib.h>
#include "sdll.h"
#include "../ompmains.h"
#include "../consts.h"

SDLLNode* sdllNodeAlloc(Env *env) {
    // SDLLNode *node = (SDLLNode*) malloc(sizeof(SDLLNode));
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

void sdllNodeUpdate(Env *env, SDLLNode *node, int32_t min_value) {
    DCHOICE(node->value, min_value, env->max_value);
}

void sdllGenerateUnsorted(Env *env, SDLL *sdll) {
    sdll->size = _choice(env->min_size, env->max_size);

    SDLLNode *current = NULL;
    for (int32_t i = 0; i < sdll->size; i++) {
        if (current == NULL) {
            sdll->head = sdllNodeAlloc(env); //sdllNodeGenerate(env);
            sdllNodeUpdate(env, sdll->head, env->min_value);
            current = sdll->head;
        } else {
            current->next = sdllNodeAlloc(env); //sdllNodeGenerate(env);
            sdllNodeUpdate(env, current->next, DACCESS(current->value));
            current->next->prev = current;
            current = current->next;
        }
    }
}


int8_t sdllIsSorted(SDLL *sdll) {
    if (sdll->size == 1) {
        // original code has the following line to force delayed execution
        // int32_t value = sdll->head->value;
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

void sdllGenerate(Env *env, SDLL *sdll) {
    sdllGenerateUnsorted(env, sdll);
    int8_t is_sorted = sdllIsSorted(sdll);
    // original code has the following line to count structures that
    // satisfy property
    _countIf(is_sorted);
    _ignoreIf(!is_sorted);
}

void sdllUdita(int32_t size) {
    int tid = omp_get_thread_num();

    SDLLNode pool[POOL_SIZE];
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
}

int main(int argc, char *argv[]) {
    return uditaMainOMP(argc, argv, (void (*)(...))sdllUdita);
}
