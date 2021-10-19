#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "ds.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void dsSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    Record array[10];
    disjset ds(array);
    ds.setPathCompression(true);
    ds.create(n);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        switch (op) {
            case 0:
            {
            int el = _choice(0, n - 1);
            if (el != INVALID_VALUE) {
                ds.find(el);
            }
            break;
            }
        case 1:
            int x = _choice(0, n - 1);
            int y = _choice(0, n - 1);
            _ignoreIf(x == y);
            if (x != INVALID_VALUE && y != INVALID_VALUE) {
                ds.unionMethod(x,y);
            }
            break;
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))dsSeqOMP);
}
