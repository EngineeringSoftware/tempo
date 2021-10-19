#ifndef FR_H
#define FR_H

#include <stdint.h>
#define MAX_VARS 2
#define MAX_MTDS 2
#define MAX_SUPER_CLS 2

typedef struct _method {
    // -, ++, --, sizeof(), +=, -=, *=, /=, %/
    // int8_t op;
    // whether to add parens
    // int8_t add_parens;
    // which variable to invoke
    int8_t var_num;
    // which class to invoke
    int8_t cls_num;
    // none, volatile, static
    int8_t qualifier;
} MTD;

typedef struct _cls {
    // none, volatile, mutable, static
    int8_t vars[MAX_VARS];
    // collection of class methods
    MTD mtds[MAX_MTDS];
    // who is the super class of this
    int8_t super_cls[MAX_SUPER_CLS];
} CLS;

typedef struct _fr_env {
    int32_t num_classes;
    int32_t num_variables;
    int32_t num_methods;
    int32_t num_supercls;
} ENV;

#endif