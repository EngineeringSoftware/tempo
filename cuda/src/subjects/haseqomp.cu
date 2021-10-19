#include <stdint.h>
#include <time.h>
#include "omp.h"

#include "ha.h"
#include "../consts.h"
#include "../explore.h"
#include "../ompmains.h"

/* @private */
void haSeqOMP(int32_t n) {
    int tid = omp_get_thread_num();

    int array[10];
    HA ha(array);
    for (int i = 0; i < n; ++i) {
        int op = _choice(0, 1);
        int value = _choice(0, n - 1);
        switch (op) {
        case 0:
            ha.insert(value);
            break;
        case 1:
            ha.remove();
            break;
        }
    }

    _countIf(1);
}

int main(int argc, char *argv[]) {
    return seqMainOMP(argc, argv, (void (*)(...))haSeqOMP);
}
