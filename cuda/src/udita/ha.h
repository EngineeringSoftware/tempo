
/*
 * This file is implementation of
 * oracle.heaparray.HeapArray
 * from the UDITA repository.
 */

#ifndef _HA_H
#define _HA_H

#include <stdint.h>
#include "../consts.h"

typedef struct _heaparray_env {
    int32_t max_array_length;
} Env;

typedef struct _heaparray {
    int32_t size;
    int32_t arraylength;
    int32_t array[POOL_SIZE];
} HeapArray;

// ----------------------------------------


#endif
