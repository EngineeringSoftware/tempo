
#include <assert.h>
#include <stdlib.h>
#include "sequtil.h"

void testQueue(void) {
    MethodSeq *ms1 = allocMS();
    MethodSeq *ms2 = allocMS();

    Queue *queue = allocQueue();
    queueAppend(queue, ms1);
    queueAppend(queue, ms2);
    assert(queueSize(queue) == 2);

    MethodSeq **sequences = queueGetSequences(queue);
    assert(sequences[0] == ms1);
    assert(sequences[1] == ms2);

    MethodSeq *ms3 = msAppend(ms1, 5, 6);
    MethodSeq *ms4 = msAppend(ms3, 10, 11);

    assert(msSize(ms1) == 0);
    assert(msSize(ms2) == 0);
    assert(msSize(ms3) == 1);
    assert(msSize(ms4) == 2);

    assert(ms3->ops[0] == 5);
    assert(ms3->vals[0] == 6);

    assert(ms4->ops[0] == 5);
    assert(ms4->vals[0] == 6);
    assert(ms4->ops[1] == 10);
    assert(ms4->vals[1] == 11);

    free(ms3);
    free(ms4);
    queueFree(queue);
    free(sequences);
}

int main(void) {
    testQueue();
    return 0;
}
