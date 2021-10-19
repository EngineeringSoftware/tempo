#ifndef NC_H
#define NC_H

#include <stdint.h>

typedef struct _nestedclasses {
    // number of classes nested
    int32_t size;
    // location of field accesses (inner or outer class)
    int8_t location;
    // static or non-static, access modifiers, ...
    int8_t field_modifiers;
    // static or non-static, access modifiers, ...
    int8_t function_modifiers;
    // how the field is accessed (::, ., ->)
    int8_t access_operator;
    // object initialization ((), {}, new, ...)
    int8_t initialization_operator;
    // which class's field will be accessed
    int8_t class_accessed;
} NC;

typedef struct _nestedclasses_env {
    int32_t size;
    int32_t num_modifiers;
    int32_t num_access_operators;
    int32_t num_initialization_operators;
} Env;

__device__ void ncGenerate(Env*, NC*);

#endif