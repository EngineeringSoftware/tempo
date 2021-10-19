
/*
 * This file is implementation of
 * oracle.nqueens.NQueens
 * from the UDITA repository.
 */

#ifndef _NQUEENS_H
#define _NQUEENS_H

#include <stdint.h>
#include "../consts.h"

typedef struct _nqueens_env {
    int8_t max_size;
} Env;

typedef struct _nqueens {
    int8_t size;
    int8_t array[NQUEENS_POOL_SIZE];
} NQueens;

// ----------------------------------------


#endif
