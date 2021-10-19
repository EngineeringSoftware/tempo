
#ifndef DELAYED_H
#define DELAYED_H

/**
 * Delayed int.
 */
typedef struct {
    int ix;
} dint;

// __device__ dint* _delayed(int min, int max);
__device__ dint _delayed(int *choices, int min, int max);

// __device__ int _force(dint *val);
__device__ int _force(int *choices, dint dix);

#endif
