#include "../explore.h"
#include "../mains.h"
#include "fr.h"

#define OUTPUT_BUFFER 1000000000

__device__ int8_t fr_buffer[OUTPUT_BUFFER];

// how many fields are in the EIS struct and nested structs
#define MTD_STRUCT_FIELDS 3
#define MAX_OPERATORS 11
#define PARENS_OPTIONS 2
#define NUM_MODIFIERS 4
#define CLS_SIZE 128
#define NUM_QUALIFIERS 3

void printFrs(int8_t *field_references, int valid_frs, int struct_size) {
    printf("Programs:\n");
    for (int i = 0; i < valid_frs; i++) {
        for (int j = 0; j < struct_size; j++) {
            printf("%d ", field_references[(i * struct_size) + j]);
        }
        printf("\n");
    }
}

__device__ void frStore(CLS cls_arr[], ENV *env) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t TOTAL_SIZE = env->num_classes * (MTD_STRUCT_FIELDS * env->num_methods + env->num_supercls + env->num_variables);
    int32_t offset = idx * TOTAL_SIZE;
    int32_t curr_ind = 0;
    
    if (offset > OUTPUT_BUFFER) {
        printf("ERROR in thread %d, index %d\n", idx, offset);
    }
    for (int step = 0; step < (env->num_classes); step++) {
        CLS& cls = cls_arr[step];
        for (int i = 0; i < (env->num_variables); i++) {
            fr_buffer[offset + curr_ind +i] = cls.vars[i];
        }

        curr_ind += env->num_variables;

        for (int i = 0; i < env->num_methods; i++) {
            // fr_buffer[offset + curr_ind] = cls.mtds[i].op;
            // fr_buffer[offset + curr_ind] = cls.mtds[i].add_parens;
            fr_buffer[offset + curr_ind ] = cls.mtds[i].var_num;
            fr_buffer[offset + curr_ind + 1] = cls.mtds[i].cls_num;
            fr_buffer[offset + curr_ind + 2] = cls.mtds[i].qualifier;
            curr_ind += 3;
        }

        for (int i = 0; i < env->num_supercls; i++) {
            fr_buffer[offset + curr_ind + i] = cls.super_cls[i];
        }
        curr_ind += env->num_supercls;
    }
}

__device__ void mtdGenerate(MTD* mtd, ENV* env, int cls_num) {
    // mtd->op = (int8_t) _choice(0, MAX_OPERATORS - 1);
    // mtd->add_parens = (int8_t) _choice(0, PARENS_OPTIONS - 1);
    mtd->var_num = (int8_t) _choice(0, env->num_variables - 1);
    mtd->cls_num = (int8_t) _choice(0, cls_num);
    mtd->qualifier = (int8_t) _choice(0, NUM_QUALIFIERS - 1);
}

__device__ void clsGenerate(CLS* cls, ENV* env, int cls_num) {
    for (int i = 0; i < env->num_variables; i++) {
        cls->vars[i] = _choice(0, NUM_MODIFIERS - 1);
    }

    for (int i = 0; i < env->num_methods; i++) {
        MTD m = {
            // .op = 1,
            // .add_parens = 0, 
            .var_num = 0,
            .cls_num = 0,
            .qualifier = 0
        };
        mtdGenerate(&m, env, cls_num);
        cls->mtds[i] = m;
    }
    
    for (int i = 0; i < env->num_supercls; i++) {
        cls->super_cls[i] = _choice(0, cls_num);
    }

    // for (int i = 0; i < env->num_supercls; i++) {
    //     for (int j = 0; j < env->num_supercls; j++){
    //         _ignoreIf((i != j) && (cls->super_cls[i] == cls->super_cls[j]));
    //     }
    // }
}

__global__ void frUdita(int32_t bck_active, int32_t size) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    ENV env = {
        .num_classes = size,
        .num_supercls = 2,
        .num_variables = 1,
        .num_methods = 1,
    };

    
    CLS cls_arr[CLS_SIZE];

    for (int j = 0; j < size; j++) {
        clsGenerate(&cls_arr[j], &env, j);
    }

    for (CLS c: cls_arr) {
        for (MTD m: c.mtds) {
            _ignoreIf(cls_arr[m.cls_num].vars[m.var_num] != 3 && m.qualifier == 2);
        }
    }

    frStore(cls_arr, &env);
    _countIf(TRUE);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Incorrect arguments: size\n");
        exit(1);
    }
    int32_t size = atoi(argv[1]);

    // establish context
    cudaFree(0);

    float time;
    cudaEvent_t start, stop;

    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    int* args = (int*) malloc(sizeof(int));
    args[0] = size;

    CUDA_CHECK_RETURN(cudaEventRecord(start));
    explore((void (*)(...)) frUdita, args, 1);
    // EXPLORE(frUdita<<<starting_blocks, starting_threads>>>(active_threads, size));
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);


    int valid_frs = *(_bck_stats->if_counter);
    int struct_size = size * (MTD_STRUCT_FIELDS * 1 + 2 + 1);
    int8_t *field_references = (int8_t*) calloc(valid_frs * struct_size, sizeof(int8_t));
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(field_references, fr_buffer, sizeof(int8_t) * valid_frs * struct_size, 0, cudaMemcpyDeviceToHost));
    printFrs(field_references, valid_frs, struct_size);
    free(field_references);

    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
}