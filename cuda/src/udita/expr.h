#ifndef EXPR_H
#define EXPR_H

#include <stdint.h>
#include "../explore.h"

enum primitive_types {
    int_type,
    const_int_type,
    bool_type,
    double_type,
    const_double_type,
};

enum assignment_operators {
    equals,
};

enum class_types {
    primary_expr,
    postfix_expr,
    init_list,
    assignment_expr,
};

template <typename T>
class Generator {
   public:
      __device__ virtual T generate(int max_size) = 0;
};

class ProgramType {
   public:
      __device__ virtual int to_string(char* output, int index) = 0;
};

class PrimaryExpr: public ProgramType {
    public:
        bool is_lit;
        bool is_this;
        bool is_id;
        bool is_lambda;

        __device__ PrimaryExpr(bool is_lit, bool is_this, bool is_id, bool is_lambda) {
            this->is_lit = is_lit;
            this->is_this = is_this;
            this->is_id = is_id;
            this->is_lambda = is_lambda;
        }

        __device__ PrimaryExpr() = default;

        __device__ int to_string(char* output, int index) {
            class_types cls_type = class_types::primary_expr;
            output[index++] = '[';
            output[index++] = '0'+cls_type;
            output[index++] = ' ';
            output[index++] = is_lit?'1':'0';
            output[index++] = ' ';
            output[index++] = is_this?'1':'0';
            output[index++] = ' ';
            output[index++]  = is_id?'1':'0';
            output[index++] = ' ';
            output[index++] = is_lambda?'1':'0';
            output[index++] = ']';
            return index;
        }
};

class PrimaryExprGenerator: public Generator<PrimaryExpr> {
    public:
         __device__ PrimaryExpr generate(int max_size) {
             if (max_size > 0) {
                int opt = _choice(1, 4);
                if (opt == 1) {
                    return PrimaryExpr(true, false, false, false);
                }
                if (opt == 2) {
                    return PrimaryExpr(false, true, false, false);
                }
                if (opt == 3) {
                    return PrimaryExpr(false, false, true, false);
                }
                if (opt == 4) {
                    return PrimaryExpr(false, false, false, true);
                }
             }
             else {
                 _ignoreIf(TRUE);
             }
         }
};

class InitializerList: public ProgramType {
    public:
        int len;
        PrimaryExpr* clauses;

        __device__ InitializerList(int len, PrimaryExpr* clauses) {
            this->len = len;
            this->clauses = clauses;
        }

        __device__ InitializerList() = default;

        __device__ int to_string(char* output, int index) {
            class_types cls_type = class_types::init_list;
            output[index++] = '[';
            output[index++] = '0'+cls_type;
            output[index++] = ' ';
            output[index++] = '0'+len;
            output[index++] = ' ';
            int tmp_ind = index;
            for (int i = 0; i < len; i++) {
                tmp_ind  = clauses[i].to_string(output, tmp_ind);
            }
            output[tmp_ind++] = ']';
            return tmp_ind;
        }
};

class InitializerListGenerator: public Generator<InitializerList> {
    public:
        PrimaryExprGenerator prim_gen; 

        __device__ InitializerListGenerator(PrimaryExprGenerator prim_gen) {
            this->prim_gen = prim_gen;
        }

        __device__ InitializerListGenerator() {
            PrimaryExprGenerator prim_gen;
            this->prim_gen = prim_gen;
        }

        __device__ InitializerList generate(int max_size) {
             if (max_size > 0) {
                PrimaryExpr prim_exprs[3];
                for (int i = 0; i < 3; i++) {
                    prim_exprs[i] = prim_gen.generate(max_size - 1);
                }
                return InitializerList(3, prim_exprs);
             }
             else {
                 _ignoreIf(TRUE);
             }
         }
};

class PostfixExpr: public ProgramType {
    public:
        bool is_prim_expr;
        bool is_expr_list;
        PrimaryExpr prim_expr;
        primitive_types simple_type;
        InitializerList expr_list;

        __device__ PostfixExpr(bool is_prim_expr, bool is_expr_list, PrimaryExpr prim_expr, primitive_types simple_type, InitializerList expr_list) {
            this->is_prim_expr = is_prim_expr;
            this->is_expr_list = is_expr_list;
            this->prim_expr = prim_expr;
            this->simple_type = simple_type;
            this->expr_list = expr_list;
        }

        __device__ int to_string(char* output, int index) {
            class_types cls_type = class_types::postfix_expr;
            output[index++] = '[';
            output[index++] = '0'+cls_type;
            output[index++] = ' ';
            output[index++] = is_prim_expr?'1':'0';
            output[index++] = ' ';
            output[index++] = is_expr_list?'1':'0';
            output[index++] = ' ';
            if (is_prim_expr) {
                int new_ind  = prim_expr.to_string(output, index++);
                output[new_ind++] = ']';
                return new_ind;
            }
            if (is_expr_list) {
                output[index++] = '0'+simple_type;
                output[index++] = ' ';
                int new_ind  = expr_list.to_string(output, index++);
                output[new_ind++] = ']';
                return new_ind++;
            }
        }
};

class PostfixExprGenerator: public Generator<PostfixExpr> {
    public:
        PrimaryExprGenerator prim_gen;
        InitializerListGenerator init_list_gen;

        __device__ PostfixExprGenerator(PrimaryExprGenerator prim_gen, InitializerListGenerator init_list_gen) {
            this->prim_gen = prim_gen;
            this->init_list_gen = init_list_gen;

        }

        __device__ PostfixExprGenerator() {
            PrimaryExprGenerator prim_gen;
            InitializerListGenerator init_list_gen;
            this->prim_gen = prim_gen;
            this->init_list_gen = init_list_gen;
        }

         __device__ PostfixExpr generate(int max_size) {
             if (max_size > 0) {
                InitializerList fake_init;
                PrimaryExpr fake_prim;
                int option = _choice(1, 2);
                if (option == 1) {
                    return PostfixExpr(true, false, prim_gen.generate(max_size - 1), primitive_types::int_type, fake_init);
                }
                if (option == 2) {
                    primitive_types prim_type = static_cast<primitive_types>(_choice(int_type, const_double_type));
                    return PostfixExpr(false, true, fake_prim, prim_type, init_list_gen.generate(max_size-1));
                }
             }
             else {
                 _ignoreIf(TRUE);
             }
         }
};

class AssignmentExpr: public ProgramType {
    public:
        bool is_primary;
        bool is_init_list;
        PrimaryExpr prim_expr;
        assignment_operators op;
        InitializerList init_list;

        __device__ AssignmentExpr(bool is_primary, bool is_init_list, PrimaryExpr prim_expr, assignment_operators op, InitializerList init_list) {
            this->is_primary = is_primary;
            this->is_init_list = is_init_list;
            this->prim_expr = prim_expr;
            this->op = op;
            this->init_list = init_list;
        }

        __device__ int to_string(char* output, int index) {
            class_types cls_type = class_types::assignment_expr;
            output[index++] = '[';
            output[index++] = '0'+cls_type;
            output[index++] = ' ';
            output[index++] = is_primary?'1':'0';
            output[index++] = ' ';
            output[index++] = is_init_list?'1':'0';
            output[index++] = ' ';
            if (is_primary) {
                int new_ind  = prim_expr.to_string(output, index++);
                output[new_ind] = ']';
                return new_ind+1;
            }
            if (is_init_list) {
                index  = prim_expr.to_string(output, index++);
                output[index++] = ' ';
                output[index++] = '0'+op;
                output[index++] = ' ';
                int new_ind  = init_list.to_string(output, index++);
                output[new_ind++] = ']';
                return new_ind;
            }
        }
};

class AssignmentExprGenerator: public Generator<AssignmentExpr> {
    public:
        PrimaryExprGenerator prim_gen;
        InitializerListGenerator init_list_gen;

        __device__ AssignmentExprGenerator(PrimaryExprGenerator prim_gen, InitializerListGenerator init_list_gen) {
            this->prim_gen = prim_gen;
            this->init_list_gen = init_list_gen;
        }

        __device__ AssignmentExprGenerator() {
            PrimaryExprGenerator prim_gen;
            InitializerListGenerator init_list_gen;
            this->prim_gen = prim_gen;
            this->init_list_gen = init_list_gen;
        }

         __device__ AssignmentExpr generate(int max_size) {
             if (max_size > 0) {
                InitializerList fake_init;
                PrimaryExpr fake_prim;

                int option = _choice(1, 2);
                if (option == 1) {
                    return AssignmentExpr(true, false, prim_gen.generate(max_size - 1), assignment_operators::equals, fake_init);
                }
                if (option == 2) {
                    return AssignmentExpr(false, true, prim_gen.generate(max_size - 1), assignment_operators::equals, init_list_gen.generate(max_size-1));
                }
             }
             else {
                 _ignoreIf(TRUE);
             }
         }
};

#endif