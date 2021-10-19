#ifndef FS_H
#define FS_H

#include <stdint.h>

#define MAX_MODIFIERS 11

typedef struct _functionspecifiers {
    int8_t *specifiers;
    int32_t index;
} FS;

__device__ void fsGenerate(FS*, int32_t);

#endif