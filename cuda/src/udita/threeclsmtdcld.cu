#include "../explore.h"
#include "../mains.h"
#include "threeclsmtdcld.h"

#define OUTPUT_BUFFER 1000000000

__device__ int8_t tci_buffer[OUTPUT_BUFFER];

// how many fields are in the FS struct
#define STRUCT_FIELDS 7

void printFss(int8_t *tci_arr, int valid_tci, int struct_size) {
    printf("Programs:\n");
    for (int i = 0; i < valid_tci; i++) {
        for (int j = 0; j < struct_size; j++) {
            printf("%d ", tci_arr[(i * struct_size) + j]);
        }
        printf("\n");
    }
}

__device__ void tciStore(TCI *tci) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t offset = idx * STRUCT_FIELDS;
    tci_buffer[offset] = tci->f_modifier; 
    tci_buffer[offset + 1] = tci->m_access_mod; 
    tci_buffer[offset + 2] = tci->f_assignment_access_type; 
    tci_buffer[offset + 3] = tci->m_rel; 
    tci_buffer[offset + 4] = tci->m_access_type; 
    tci_buffer[offset + 5] = tci->subcls_rel; 
    tci_buffer[offset + 6] = tci->supercls_rel; 
}

__device__ void tciGenerate(TCI *tci) {
    tci->f_modifier = (int8_t) _choice(0, MAX_FIELD_MODIFIER - 1);
    tci->m_access_mod = (int8_t) _choice(0, MAX_MTD_MODIFIER - 1);
    tci->f_assignment_access_type = (int8_t) _choice(0, MAX_FIELD_ACCESS_TYPE - 1);
    tci->m_rel = (int8_t) _choice(0, MAX_MTD_REL - 1);
    tci->m_access_type = (int8_t) _choice(0, MAX_MTD_ACCESS_TYPE - 1);
    tci->subcls_rel = (int8_t) _choice(0, MAX_SUBCLASS_REL - 1);
    tci->supercls_rel = (int8_t) _choice(0, MAX_SUPERCLASS_REL - 1);
}

__global__ void threeClsMtdCldUdita(int32_t bck_active, int32_t size) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    TCI tci = {
        .f_modifier = 0,
        .m_access_mod = 0,
        .f_assignment_access_type = 0,
        .m_rel = 0,
        .m_access_type = 0,
        .subcls_rel = 0,
        .supercls_rel = 0,
    };

    tciGenerate(&tci);
    tciStore(&tci);
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

    explore((void (*)(...)) threeClsMtdCldUdita, args, 1);
    // EXPLORE(threeClsMtdCldUdita<<<starting_blocks, starting_threads>>>(active_threads, size));
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);

    int valid_tci = *(_bck_stats->if_counter);
    int8_t *tci_arr = (int8_t*) calloc(valid_tci * STRUCT_FIELDS, sizeof(int8_t));
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(tci_arr, tci_buffer, sizeof(int8_t) * valid_tci * STRUCT_FIELDS, 0, cudaMemcpyDeviceToHost));
    
    printFss(tci_arr, valid_tci, STRUCT_FIELDS);

    free(tci_arr);

    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
}