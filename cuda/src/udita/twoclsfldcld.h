#define MAX_FIELD_TYPE 5
#define MAX_FIELD_ACCESS_MOD 3
#define IS_STATIC_CHOICE 2
#define MAX_FIELD_ACCESS_TYPE 6
#define MAX_SUBCLS_REL 3


typedef struct _double_cls_info {
    int8_t f_type;
    int8_t f_access_mod;
    int8_t is_static; 
    int8_t f_access_type; 
    int8_t subcls_rel; 
} DCI;
