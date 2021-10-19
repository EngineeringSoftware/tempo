#include "../explore.h"
#include "../mains.h"
#include "expr.h"

#define OUTPUT_BUFFER 1000000000

// how many fields are in the FS struct
#define STRUCT_FIELDS 11

__global__ void exprUdita(int32_t bck_active, int32_t size) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    AssignmentExprGenerator gen;
    AssignmentExpr expr = gen.generate(size);
    char output[500];
    expr.to_string(output, 0);

    printf("%s\n", output);
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
    printf("Programs: \n");
    explore((void (*)(...)) exprUdita, args, 1);
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