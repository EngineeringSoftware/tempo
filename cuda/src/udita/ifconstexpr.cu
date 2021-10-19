#include "../explore.h"
#include "../mains.h"
#include "ifconstexpr.h"

#define OUTPUT_BUFFER 1000000000

__device__ int8_t nce_buffer[OUTPUT_BUFFER];

// how many fields are in the DCI struct
#define STRUCT_FIELDS 1

void printNce(int8_t *nce_arr, int valid_nce, int struct_size) {
    printf("Programs:\n");
    for (int i = 0; i < valid_nce; i++) {
        for (int j = 0; j < struct_size; j++) {
            printf("%d ", nce_arr[(i * struct_size) + j]);
        }
        printf("\n");
    }
}

__device__ void nceStore(NCE *nce, int32_t size) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t offset = idx * (STRUCT_FIELDS + 2 * size);
    int32_t curr_ind = 0;

    for (int i = 0; i < size; i++) {
        nce_buffer[offset + i] = nce->check_cond[i];
    }

    curr_ind += size;

    nce_buffer[offset + curr_ind] = nce->base_var_type;

    curr_ind++;

    for (int i = 0; i < size; i++) {
        nce_buffer[offset + curr_ind + i] = nce->indent_levels[i];
    }
}

__device__ void nceGenerate(NCE *nce, int32_t size) {
    for (int i = 0; i < size; i++) {
        nce->check_cond[i] = _choice(0, MAX_CHECK_COND - 1);
        nce->indent_levels[i] = _choice(0, MAX_IF_COND - 1);
    }
    nce->base_var_type = _choice(0, MAX_TYPES - 1);
}

__global__ void nceUdita(int32_t bck_active, int32_t size) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    NCE nce = {
        .base_var_type = 0,
    };

    nceGenerate(&nce, size);
    nceStore(&nce, size);
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

    CUDA_CHECK_RETURN(cudaEventRecord(start));
    int* args = (int*) malloc(sizeof(int));
    args[0] = size;

    explore((void (*)(...)) nceUdita, args, 1);
    // EXPLORE(twoClsMtdCldUdita<<<starting_blocks, starting_threads>>>(active_threads, size));
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);

    int valid_nce = *(_bck_stats->if_counter);
    int struct_size = STRUCT_FIELDS + 2 * size;
    int8_t *nce_arr = (int8_t*) calloc(valid_nce * struct_size, sizeof(int8_t));
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(nce_arr, nce_buffer, sizeof(int8_t) * valid_nce * struct_size, 0, cudaMemcpyDeviceToHost));
    
    printNce(nce_arr, valid_nce, struct_size);

    free(nce_arr);

    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
}