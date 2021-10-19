
#ifndef SEQUTIL_H
#define SEQUTIL_H

#define MAXOPS 100

typedef struct _methodseq {
    /* index for next operation and value */
    int ix;
    /* operations */
    int ops[MAXOPS];
    /* values */
    int vals[MAXOPS];
    /* next method sequence */
    struct _methodseq *next;
} MethodSeq;

typedef struct _queue {
    int size;
    MethodSeq *head;
} Queue;

/* ---------------------------------------- */

MethodSeq* allocMS(void);
int msSize(MethodSeq *ms);
MethodSeq* msAppend(MethodSeq *ms, int op, int val);
void msPrint(MethodSeq *ms);

/* ---------------------------------------- */

Queue* allocQueue(void);
void queueFree(Queue *queue);
void queueAppend(Queue *queue, MethodSeq *ms);
int queueSize(Queue *queue);
MethodSeq** queueGetSequences(Queue *queue);

/* ---------------------------------------- */

__host__ int seqMainCPU(int32_t n, void (*lambda)(MethodSeq*));

#endif
