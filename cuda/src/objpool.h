
#ifndef OBJPOOL_H
#define OBJPOOL_H

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "consts.h"
#include "explore.h"

/* Flag to include NULL value in the pool */
#define INCLUDE_NULL 1

/* Flag to exclude NULL value from the pool */
#define EXCLUDE_NULL 0

/* Used for size when initializing an infinite object pool */
#define INFINITE_SIZE -1

#define DefObjPool(TYPE) \
                              \
    typedef struct _objpool ## TYPE { \
        /* pool */                              \
        TYPE objects[POOL_SIZE];                 \
        /* flag if this pool infinite */        \
        int8_t infinite;                        \
        /* size of the pool */                  \
        int32_t size;                           \
        /* current position in the object pool */       \
        int32_t index;                                  \
        /* flag if this pool includes null */           \
        int8_t include_null;                            \
        /* public functions */                          \
        TYPE* (*getAny)(struct _objpool ## TYPE *op);            \
        TYPE* (*getNew)(struct _objpool ## TYPE *op);            \
        TYPE* (*getObject)(struct _objpool ## TYPE *op, int32_t);            \
    } TYPE ## Pool;                                             \
                                                                \
    DEVICE void init ## TYPE ## Pool(TYPE ## Pool *op, int32_t size, int8_t include_null); \
                                                                        \
    DEVICE void opExpandPool ## TYPE(TYPE ## Pool *op);                          \
                                                                        \
    DEVICE TYPE* opGetObject ## TYPE(TYPE ## Pool *op, int32_t ix);               \
                                                                        \
    DEVICE TYPE* opGetAny ## TYPE(TYPE ## Pool *op);                              \
                                                                        \
    DEVICE TYPE* opGetNew ## TYPE(TYPE ## Pool *op);                              \
                                                                        \
    extern int32_t __variable__ ## TYPE

#define ImpObjPool(TYPE) \
                         \
    DEVICE void init ## TYPE ## Pool(TYPE ## Pool *op, int32_t size, int8_t include_null) { \
        /* set functions */                                             \
        op->getAny = &opGetAny ## TYPE;                                         \
        op->getNew = &opGetNew ## TYPE;                                         \
        op->getObject = &opGetObject ## TYPE;                                         \
                                                                        \
        op->infinite = (size == INFINITE_SIZE);                         \
        op->size = size + (include_null ? 1 : 0);                       \
        op->include_null = include_null;                                \
        op->index = (include_null ? 0 : -1);                            \
                                                                        \
        /* make sure that objects are all 0 initialized */              \
        memset(op->objects, 0, sizeof(TYPE) * POOL_SIZE);               \
                                                                        \
        assert((size == INFINITE_SIZE) || (size >= 0 && size <= POOL_SIZE)); \
        if (include_null) {                                             \
            /* (we do not keep pointers so we assume at index 0 is NULL) */ \
            /* op->objects[0] = NULL; */ \
        }                                                       \
        if (!op->infinite) {                                    \
            /* create objects (we do no work here as we preallocate */  \
            /* specific size for the pool) */                           \
        }                                                               \
    }                                                                   \
                                                                        \
    DEVICE void opExpandPool ## TYPE(TYPE ## Pool *op) {                         \
        /* we are not allocating new element here as we preallocate, but */ \
        /* we check size */                                             \
        if (op->index >= POOL_SIZE) {                                   \
            printf("ERROR: not enough space in the pool %d %d\n", op->index, op->size); \
        }                                                               \
    }                                                                   \
                                                                        \
    DEVICE TYPE* opGetObject ## TYPE(TYPE ## Pool *op, int32_t ix) {              \
        if (op->infinite && (op->index == ix)) {                        \
            opExpandPool ## TYPE(op);                                           \
        }                                                               \
        /* this condition differs from Java version, because here we have */ \
        /* to explicitly return null for index 0 (if null is in the pool); */ \
        /* we create an array of objects in the pool and not pointers. */ \
        if (ix == 0 && op->include_null) {                              \
            return NULL;                                                \
        } else {                                                        \
            return &(op->objects[ix]);                                  \
        }                                                               \
    }                                                                   \
                                                                        \
    DEVICE TYPE* opGetAny ## TYPE(TYPE ## Pool *op) {                             \
        _ignoreIf(!(op->infinite) && (op->size == 0));                  \
                                                                        \
        int32_t ix;                                                     \
        if (op->infinite) {                                             \
            ix = _choice(0, op->index + 1);                             \
        } else {                                                        \
            ix = _choice(0, min(op->index + 1, op->size - 1));          \
        }                                                               \
                                                                        \
        if (ix == op->index + 1) {                                      \
            op->index++;                                                \
        }                                                               \
        return opGetObject ## TYPE(op, ix);                                     \
    }                                                                   \
                                                                        \
    DEVICE TYPE* opGetNew ## TYPE(TYPE ## Pool *op) {                             \
        _ignoreIf(!(op->infinite) && ((op->index + 1 == op->size) || (op->size == 0))); \
        return opGetObject ## TYPE(op, ++(op->index));                          \
    }                                                                   \
                                                                        \
    int32_t __variable__ ## TYPE

#define ObjPool(TYPE) \
    DefObjPool(TYPE); \
    ImpObjPool(TYPE)

#endif
