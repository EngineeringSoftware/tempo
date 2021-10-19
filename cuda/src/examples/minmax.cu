
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../errors.h"
#include "../explore.h"

__global__ void minmax(void) {
    int i = _choice(3, 3);
    int j = _choice(4, 4);
    printf("%d %d\n", i, j);
}

int main(void) {
    EXPLORE(minmax<<<starting_blocks, starting_threads>>>());
}
