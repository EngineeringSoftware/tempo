#define MAX_NESTING 128
#define MAX_TYPES 3
#define MAX_IF_COND 3
#define MAX_CHECK_COND 5

typedef struct nested_constexpr {
    // 0 - is_integral, 1 - is_void, 2 - is_floating_pt, 3 - is_class, 4 - is_scaler
    int8_t check_cond[MAX_NESTING];

    // 0 - float, 1 - int, 2 - char
    int8_t base_var_type;

    // 0 - indent level (if statement), 1 - elif, 2 - else/(out level)
    int8_t indent_levels[MAX_NESTING];
} NCE;
