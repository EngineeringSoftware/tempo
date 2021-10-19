#define MAX_FIELD_MODIFIER 2
#define MAX_MTD_MODIFIER 2
#define MAX_FIELD_ACCESS_TYPE 6
#define MAX_MTD_REL 2
#define MAX_MTD_ACCESS_TYPE 4
#define MAX_SUBCLASS_REL 3

typedef struct _double_cls_info {
    // same level, local, none
    int8_t f_modifier;
    int8_t m_access_mod;
    int8_t f_assignment_access_type;
    int8_t m_rel;
    int8_t m_access_type;
    int8_t subcls_rel;
} DCI;
