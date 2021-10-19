#define MAX_FIELD_TYPE 4
#define MAX_FIELD_ACCESS 5
#define MAX_SUBFIELD_LOC 2
#define MAX_SUBCLS_OPTIONS 2
#define MAX_SUBCLS_REL 3


typedef struct _double_cls_info {
    // int, bool, char, float
    int8_t f_type;
    // this, class this, super, instance, static, simple
    int8_t f_access;
    // bool
    int8_t subfield_loc;
    // bool
    int8_t subclass_or_not;
    // outer, inner, method inner
    int8_t subclass_rel;
} DCI;
