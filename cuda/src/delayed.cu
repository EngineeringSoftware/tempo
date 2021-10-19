
#include "delayed.h"
#include "explore.h"

__device__ dint _delayed(int *choices, int min, int max) {
    int ix = choices[0] * 4 + 1;
    choices[ix] = 0; // initialized
    choices[ix+1] = min; // min
    choices[ix+2] = max; // max
    choices[ix+3] = 0; // current value
    (choices[0])++;
    return { ix };
}

__device__ int _force(int *choices, dint dix) {
    if (!(choices[dix.ix])) {
        choices[dix.ix] = 1;
        choices[dix.ix + 3] = _choice(choices[dix.ix+1], choices[dix.ix+2]);
    }
    return choices[dix.ix + 3];
}
