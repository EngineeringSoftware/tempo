#define MAX_FIELD_MODIFIER 3
#define MAX_MTD_MODIFIER 3
#define MAX_FIELD_ACCESS_TYPE 5
#define MAX_MTD_REL 2
#define MAX_MTD_ACCESS_TYPE 4
#define MAX_SUBCLASS_REL 3
#define MAX_SUPERCLASS_REL 2

typedef struct _double_cls_info {
    // protected, public (float field)
    int8_t f_modifier;

    // private, public
    int8_t m_access_mod; 

    int8_t f_assignment_access_type;

    // this, class this, super, instance, static, simple
    int8_t m_rel;

    int8_t m_access_type;

    int8_t subcls_rel;

    int8_t supercls_rel;
} DCI;
