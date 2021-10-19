
#include <assert.h>
#include "explore.h"
#include "objpool.h"

typedef struct _tmp {
    int8_t placeholder;
} Tmp;

typedef struct _tmp2 {
    int8_t placeholder;
} Tmp2;

ObjPool(Tmp);
ObjPool(Tmp2);

__global__ void test_no_null_value(int32_t bck_active) {
    TmpPool op;
    initTmpPool(&op, 1, FALSE);

    Tmp *tmp1 = op.getNew(&op);
    Tmp *tmp2 = op.getAny(&op);
    assert(tmp1 != NULL);
    assert(tmp2 != NULL);
}

__global__ void test_limited_pool(int32_t bck_active) {
    TmpPool op;
    initTmpPool(&op, 1, FALSE);

    Tmp *tmp1 = op.getNew(&op);
    assert(tmp1 != NULL);

    // should exit here
    Tmp *tmp2 = op.getNew(&op);
    assert(FALSE);
}

__global__ void test_infinite_pool(int32_t bck_active) {
    TmpPool op;
    initTmpPool(&op, INFINITE_SIZE, FALSE);

    Tmp *tmp1 = op.getAny(&op);
    assert(tmp1 != NULL);

    Tmp *tmp2 = op.getNew(&op);
    assert(tmp2 != NULL);
}

__global__ void test_null_value(int32_t bck_active) {
    TmpPool op;
    initTmpPool(&op, 0, TRUE);

    Tmp *tmp1 = op.getAny(&op);
    assert(tmp1 == NULL);
}

__global__ void test_udita_paper_example(int32_t bck_active) {
    TmpPool op;
    initTmpPool(&op, 3, FALSE);

    Tmp *n1 = op.getNew(&op);
    Tmp *a1 = op.getAny(&op);
    Tmp *a2 = op.getAny(&op);
    Tmp *a3 = op.getAny(&op);
    Tmp *n2 = op.getNew(&op);
    Tmp *n3 = op.getNew(&op);

    assert(n1 != n2);
    assert(n1 != n3);
    assert(n2 != n3);

    assert(a1 == n1);
    assert(a2 == n1);
    assert(a3 == n1);
}

int main(int argc, char *argv[]) {
    EXPLORE(test_no_null_value<<<starting_blocks, starting_threads>>>(active_threads));
    EXPLORE(test_null_value<<<starting_blocks, starting_threads>>>(active_threads));
    EXPLORE(test_limited_pool<<<starting_blocks, starting_threads>>>(active_threads));
    EXPLORE(test_infinite_pool<<<starting_blocks, starting_threads>>>(active_threads));
    EXPLORE(test_udita_paper_example<<<starting_blocks, starting_threads>>>(active_threads));
}
