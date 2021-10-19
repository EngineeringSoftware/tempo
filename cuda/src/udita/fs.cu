#include "../explore.h"
#include "../mains.h"
#include "fs.h"

#define OUTPUT_BUFFER 1000000000

__device__ int8_t fs_buffer[OUTPUT_BUFFER];

// how many fields are in the FS struct
#define STRUCT_FIELDS 11

void printFss(int8_t *function_specifiers, int valid_fss, int struct_size) {
    printf("Programs:\n");
    for (int i = 0; i < valid_fss; i++) {
        for (int j = 0; j < struct_size; j++) {
            printf("%d ", function_specifiers[(i * struct_size) + j]);
        }
        printf("\n");
    }
}

__device__ void fsStore(FS *fs) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t offset = idx * fs->index;

    for (int i = 0; i < fs->index; i++) {
        fs_buffer[offset + i] = fs->specifiers[i];
    }
}

__device__ void fsGenerate(FS *fs, int32_t size) {
    for (int i = 0; i < size; i++) {
        fs->specifiers[fs->index++] = (int8_t) _choice(0, MAX_MODIFIERS);
    }
}

__global__ void fsUdita(int32_t bck_active, int32_t size) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    int8_t specifiers[MAX_MODIFIERS];
    for (int i = 0; i < MAX_MODIFIERS; i++) {
        specifiers[i] = 0;
    }

    FS fs = {
        .specifiers = specifiers,
        .index = 0,
    };

    fsGenerate(&fs, size);
    // fsStore(&fs);
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
    explore((void (*)(...)) fsUdita, args, 1);
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);

    // int valid_fss = *(_bck_stats->if_counter);
    // int8_t *function_specifiers = (int8_t*) calloc(valid_fss * size, sizeof(int8_t));
    // CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(function_specifiers, fs_buffer, sizeof(int8_t) * valid_fss * size, 0, cudaMemcpyDeviceToHost));
    
    // printFss(function_specifiers, valid_fss, size);

    // free(function_specifiers);

    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
    free(args);
}