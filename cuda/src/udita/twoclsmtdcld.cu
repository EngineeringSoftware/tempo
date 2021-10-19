#include "../explore.h"
#include "../mains.h"
#include "twoclsmtdcld.h"

#define OUTPUT_BUFFER 1000000000

__device__ int8_t dci_buffer[OUTPUT_BUFFER];

// how many fields are in the DCI struct
#define STRUCT_FIELDS 6

void printDci(int8_t *dci_arr, int valid_dci, int struct_size) {
    printf("Programs:\n");
    for (int i = 0; i < valid_dci; i++) {
        for (int j = 0; j < struct_size; j++) {
            printf("%d ", dci_arr[(i * struct_size) + j]);
        }
        printf("\n");
    }
}

__device__ void dciStore(DCI *dci) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t offset = idx * STRUCT_FIELDS;
    dci_buffer[offset] = dci->f_modifier; 
    dci_buffer[offset + 1] = dci->m_access_mod; 
    dci_buffer[offset + 2] = dci->f_assignment_access_type; 
    dci_buffer[offset + 3] = dci->m_rel; 
    dci_buffer[offset + 4] = dci->m_access_type; 
    dci_buffer[offset + 5] = dci->subcls_rel; 
}

__device__ void dciGenerate(DCI *dci) {
    dci->f_modifier = (int8_t) _choice(0, MAX_FIELD_MODIFIER - 1);
    dci->m_access_mod = (int8_t) _choice(0, MAX_MTD_MODIFIER - 1);
    dci->f_assignment_access_type = (int8_t) _choice(0, MAX_FIELD_ACCESS_TYPE - 1);
    dci->m_rel = (int8_t) _choice(0, MAX_MTD_REL - 1);
    dci->m_access_type = (int8_t) _choice(0, MAX_MTD_ACCESS_TYPE - 1);
    dci->subcls_rel = (int8_t) _choice(0, MAX_SUBCLASS_REL - 1);
}

__global__ void twoClsMtdCldUdita(int32_t bck_active, int32_t size) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= bck_active) {
        return;
    }

    DCI dci = {
        .f_modifier = 0,
        .m_access_mod = 0,
        .f_assignment_access_type = 0,
        .m_rel = 0,
        .m_access_type = 0,
        .subcls_rel = 0,
    };

    dciGenerate(&dci);
    dciStore(&dci);
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

    explore((void (*)(...)) twoClsMtdCldUdita, args, 1);
    // EXPLORE(twoClsMtdCldUdita<<<starting_blocks, starting_threads>>>(active_threads, size));
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Driver time: %.2lf\n", time);

    int valid_dci = *(_bck_stats->if_counter);
    int8_t *dci_arr = (int8_t*) calloc(valid_dci * STRUCT_FIELDS, sizeof(int8_t));
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(dci_arr, dci_buffer, sizeof(int8_t) * valid_dci * STRUCT_FIELDS, 0, cudaMemcpyDeviceToHost));
    
    printDci(dci_arr, valid_dci, STRUCT_FIELDS);

    free(dci_arr);

    // the following line if we use cuda-memcheck --leak-check
    cudaDeviceReset();
}