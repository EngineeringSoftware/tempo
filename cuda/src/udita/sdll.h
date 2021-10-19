
// This file is implementation of
// oracle.sorteddll.SortedDoublyLinkedList
// from the UDITA repository.

#ifndef _SDLL_H
#define _SDLL_H

#include <stdint.h>

typedef struct _dint {
    int32_t max;
    int32_t min;
    int8_t accessed;
    int32_t value;
} DInt;

#define DCHOICE(x, min_value, max_value) (x.value = _choice(min_value, max_value))
#define DACCESS(x) (x.value)

//#define DCHOICE(x, min_value, max_value) (x.max = max_value, x.min = min_value, x.accessed = 0)
//#define DACCESS(x) (x.accessed ? x.value : (x.accessed = 1, x.value = _choice(x.min, x.max), x.value))

typedef struct _sdllnode {
    DInt value;
    struct _sdllnode *next;
    struct _sdllnode *prev;
} SDLLNode;

typedef struct _sdll {
    SDLLNode *head;
    int32_t size;
} SDLL;

typedef struct {
    // min value in each node
    int32_t min_value;
    // max value in each node
    int32_t max_value;
    // min size of the list
    int32_t min_size;
    // max size of the list
    int32_t max_size;
    // index of the next object in the pool
    int8_t pix;
    // pool
    SDLLNode *pool;
} Env;

#endif
