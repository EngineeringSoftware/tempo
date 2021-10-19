#include "../explore.h"
#include "../mains.h"

#define OUTPUT_BUFFER 100000000

__device__ int32_t va_buffer[OUTPUT_BUFFER];

void printVas(int32_t *va_ints, int32_t valid_vas) {
    printf("Programs:\n");
    for (int i = 0; i < valid_vas; i++) {
        printf("%d ", va_ints[i]);
        printf("\n");
    }
}

__device__ void vaStore(int32_t val) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    va_buffer[idx] = val;
}

__global__ void vaUdita(int32_t bck_active, int32_t max_val) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }
    int32_t val = (int32_t) _choice(0, max_val);
    vaStore(val);
    _countIf(TRUE);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Incorrect arguments: max val\n");
        exit(1);
    }
    int32_t max_val = atoi(argv[1]);

    // establish context
    cudaFree(0);

    float time;
    cudaEvent_t start, stop;

    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    CUDA_CHECK_RETURN(cudaEventRecord(start));
    
    int* args = (int*) malloc(sizeof(int));
    args[0] = max_val;

    explore((void (*)(...)) vaUdita, args, 1);
    // EXPLORE(vaUdita<<<starting_blocks, starting_threads>>>(active_threads, max_val));
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);

    int valid_vas = *(_bck_stats->if_counter);
    int32_t *va_ints = (int32_t*) calloc(valid_vas, sizeof(int32_t));
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(va_ints, va_buffer, sizeof(int32_t) * valid_vas, 0, cudaMemcpyDeviceToHost));
    
    printVas(va_ints, valid_vas);

    free(va_ints);

    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
}