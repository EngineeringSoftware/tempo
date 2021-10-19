
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../errors.h"
#include "../explore.h"
#include "../delayed.h"

__global__ void minmax(void) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int choices[100] = { 0 };

    dint i = _delayed(choices, 1, 2);
    dint j = _delayed(choices, 3, 4);

    // note that the order of _force invocations is not defined in C.
    printf("%d %d %d\n", idx, _force(choices, i), _force(choices, j));
    printf("%d %d %d\n", idx, _force(choices, i), _force(choices, j));
}

int main(void) {
    EXPLORE(minmax<<<starting_blocks, starting_threads>>>());
}
