#include "../explore.h"
#include "../mains.h"
#include "nc.h"

#define OUTPUT_BUFFER 1000000000

__device__ int8_t nc_buffer[OUTPUT_BUFFER];

// how many fields are in the NC struct
#define STRUCT_FIELDS 7

void printNcs(int8_t *nested_classes, int valid_ncs, int struct_size) {
    printf("Programs:\n");
    for (int i = 0; i < valid_ncs; i++) {
        for (int j = 0; j < struct_size; j++) {
            printf("%d ", nested_classes[(i * struct_size) + j]);
        }
        printf("\n");
    }
}

__device__ void ncStore(NC *nc) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t offset = idx * STRUCT_FIELDS;

    nc_buffer[offset] = nc->size;
    nc_buffer[offset + 1] = nc->location;
    nc_buffer[offset + 2] = nc->field_modifiers;
    nc_buffer[offset + 3] = nc->function_modifiers;
    nc_buffer[offset + 4] = nc->access_operator;
    nc_buffer[offset + 5] = nc->initialization_operator;
    nc_buffer[offset + 6] = nc->class_accessed;

}

__device__ void ncGenerate(Env *env, NC *nc) {
    nc->location = (int8_t) _choice(0, env->size - 1);
    nc->field_modifiers = (int8_t) _choice(0, env->num_modifiers - 1);
    nc->function_modifiers = (int8_t) _choice(0, env->num_modifiers - 1);
    nc->access_operator = (int8_t) _choice(0, env->num_access_operators - 1);
    nc->initialization_operator = (int8_t) _choice(0 , env->num_initialization_operators - 1);
    nc->class_accessed = (int8_t) _choice(0, env->size - 1);
}

__global__ void ncUdita(int32_t bck_active, int32_t size) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    Env env = {
        .size = size,
        .num_modifiers = 2, // static or non-static
        .num_access_operators = 4, // no operator, ::, ., ->
        .num_initialization_operators = 5,// no operator, new N(), N(), N{}, N
    };

    NC nc = {
        .size = size
    };

    ncGenerate(&env, &nc);
    // ncStore(&nc);
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
    explore((void (*)(...)) ncUdita, args, 1);
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);

    // int valid_ncs = *(_bck_stats->if_counter);
    // int8_t *nested_classes = (int8_t*) calloc(valid_ncs * STRUCT_FIELDS, sizeof(int8_t));
    // CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(nested_classes, nc_buffer, sizeof(int8_t) * valid_ncs * STRUCT_FIELDS, 0, cudaMemcpyDeviceToHost));
    
    // printNcs(nested_classes, valid_ncs, STRUCT_FIELDS);

    // free(nested_classes);

    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
    free(args);
}