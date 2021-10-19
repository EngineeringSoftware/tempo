#include "omp.h"
#include "time.h"
#include <stdlib.h>
#include <stdio.h>
#include "sequtil.h"

Queue* allocQueue(void) {
    Queue *queue = (Queue*) malloc(sizeof(Queue));
    queue->size = 0;
    queue->head = NULL;
    return queue;
}

void queueFree(Queue *queue) {
    MethodSeq *prev = NULL;
    MethodSeq *current = queue->head;
    while (current != NULL) {
        prev = current;
        current = current->next;
        free(prev);
    }
    free(queue);
}

void queueAppend(Queue *queue, MethodSeq *ms) {
    (queue->size)++;
    if (queue->head == NULL) {
        queue->head = ms;
    } else {
        MethodSeq *current = queue->head;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = ms;
    }
}

int queueSize(Queue *queue) {
    return queue->size;
}

MethodSeq** queueGetSequences(Queue *queue) {
    MethodSeq **sequences = (MethodSeq**) malloc(queue->size * sizeof(MethodSeq*));
    MethodSeq *current = queue->head;

    int ix = 0;
    while (current != NULL) {
        sequences[ix++] = current;
        current = current->next;
    }
    return sequences;
}

MethodSeq* allocMS(void) {
    MethodSeq *ms = (MethodSeq*) malloc(sizeof(MethodSeq));
    ms->ix = 0;
    ms->next = NULL;
    return ms;
}

int msSize(MethodSeq *ms) {
    return ms->ix;
}

MethodSeq* msAppend(MethodSeq *ms, int op, int val) {
    MethodSeq *n = allocMS();
    n->ix = ms->ix;
    n->next = NULL;
    memcpy(n->ops, ms->ops, ms->ix * sizeof(int));
    memcpy(n->vals, ms->vals, ms->ix * sizeof(int));

    n->ops[n->ix] = op;
    n->vals[n->ix] = val;
    (n->ix)++;

    return n;
}

void msPrint(MethodSeq *ms) {
    printf("ms: ");
    for (int i = 0; i < ms->ix; i++) {
        printf("(%d %d)", ms->ops[i], ms->vals[i]);
    }
    printf("\n");
}

__host__ int seqMainCPU(int32_t n, void (*lambda)(MethodSeq*)) {
    int total_executed = 0;

    clock_t start = clock();

    Queue *to_explore = allocQueue();
    queueAppend(to_explore, allocMS());
    for (int i = 0; i < n; ++i) {
        Queue *next_to_explore = allocQueue();
        MethodSeq **sequences = queueGetSequences(to_explore);
        #pragma omp parallel for num_threads(144) collapse(3)
        for (int j = 0; j < queueSize(to_explore); j++) {
            for (int op = 0; op <= 1; op++) {
                for (int val = 0; val <= n - 1; val++) {
                    MethodSeq *ms = sequences[j];
                    MethodSeq *new_ms = msAppend(ms, op, val);
                    if (i == n - 1) {
                        lambda(new_ms);
                        #pragma omp atomic
                        total_executed++;
                    }
                    #pragma omp critical
                    queueAppend(next_to_explore, new_ms);
                }
            }
        }
        free(sequences);
        queueFree(to_explore);
        to_explore = next_to_explore;
    }

    clock_t stop = clock();
    float time_spent = (float)(stop - start) * 1000 / CLOCKS_PER_SEC;
    queueFree(to_explore);
    printf("# if_counter: %d\n", total_executed);
    printf("Driver time: %.2lf\n", time_spent);
    return 0;
}
