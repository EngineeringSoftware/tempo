
#include <stdlib.h>
#include <stdio.h>
#include "sdll.h"
#include "../consts.h"

/* EXPERIMENTAL: This program generates values for all choices for all threads */

/* Contains values */
int total[10000000];
int total_ix;
/* Contains indexes; one element is one index in total */
int labels[10000000];
int labels_ix;

int vec[100000 * 3];
int vec_length;
int choice_ix;
int counter;

#define SIZE 11

#define SPLIT(min, max, ...) for (int32_t val = min; val <= max; val++) { push(val); __VA_ARGS__ ; pop(); }

#define STOP_IF(cond) if (cond) { append(vec); run(vec); return; }

int _choice(int min, int max) { return vec[++choice_ix]; }
void _countIf(int8_t cond) { if (cond) counter++; }

SDLLNode* sdllNodeAlloc(Env *env) {
    SDLLNode *node = &(env->pool[env->pix]);
    (env->pix)++;
    if (env->pix >= POOL_SIZE) {
        printf("ERROR: not enough objects in the pool\n");
    }

    //    node->value = { 0 };
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

int8_t sdllIsSorted(SDLL *sdll) {
    if (sdll->size == 1) {
        // original code has the following line to force delayed execution
        int32_t value = DACCESS(sdll->head->value);
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
    _countIf(is_sorted);
    // _ignoreIf(!is_sorted);
}

void run(int *vec) {
    choice_ix = 0;
    SDLLNode pool[POOL_SIZE] = { 0 };
    Env env = {
        .min_value = 0,
        .max_value = /*size*/SIZE - 1,
        .min_size = 0,
        .max_size = /*size*/SIZE,
        .pix = 0,
        .pool = pool,
    };
    SDLL sdll = {
        .head = NULL,
        .size = 0,
    };

    sdllGenerate(&env, &sdll);
}

void append(int *vec) {
    labels[labels_ix++] = total_ix;
    for (int i = 0; i < vec_length; i++) {
        total[total_ix++] = vec[i+1];
    }
}

void printTotal(void) {
    for (int i = 0; i < labels_ix; i++) {
        printf("%d, ", labels[i]);
    }
    printf("\n");
    for (int i = 0; i < total_ix; i++) {
        printf("%d, ", total[i]);
    }
    printf("\n");
}

void push(int val) {
    vec[++vec_length] = val;
}

void pop() {
    vec_length--;
}

/* @key */
void values(int size, int min_value, int max_value) {
    STOP_IF(size <= 0)
    SPLIT(min_value, max_value, { values(size - 1, val, max_value); })
}

/* @key */
int main(int argc, char *argv[]) {
    if (argc <= 1) {
        printf("Please provide max size\n");
        return 1;
    }

    int min_size = 0;
    int max_size = atoi(argv[1]);
    SPLIT(min_size, max_size, { values(val, 0, max_size - 1); })
    // printTotal();
    printf("%d\n", counter);

    return 0;
}
