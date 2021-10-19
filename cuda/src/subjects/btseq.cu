
#include "bt.h"
#include "sequtil.h"
#include "../mains.h"
#include "../consts.h"

/* @private */
__host__ void btSeqCPU(MethodSeq *ms) {
    Node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    BT bt(&env);
    for (int i = 0; i < msSize(ms); ++i) {
        if (ms->ops[i] == 0) {
            bt.add(ms->vals[i]);
        } else {
            bt.remove(ms->vals[i]);
        }
    }
}

/* @private */
__device__ void btDriver(const int32_t n, const int32_t print_id) {
    Node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);
    
    BT bt(&env);
    for (int i = 0; i < n; ++i) {
        const int op = _choice(0, 1);
        const int value = _choice(0, n - 1);
        if (op == 0) {
            bt.add(value);
        } else {
            bt.remove(value);
        }
    }

    _countIf(1);

    #ifdef BTSEQ_DEBUG
    // if (tid == print_id) {
    //     bt.print();
    // }
    #endif

    #ifdef RUN_TEST
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    bt.add(tid);
    #endif
}

__global__ void btDynamic(int32_t n, int32_t *choices, int32_t total, int index, int min, int max);

__device__ int _choiceDynamic(int n, int *choices, int *total, int *index, int min, int max)  {
    if (*index < *total) {
        int val = choices[*index];
        (*index)++;
        return val;
    } else {
        btDynamic<<<1, max - min + 1>>>(n, choices, *total + 1, 0, min, max);
        asm("exit;");
        return -1;
    }
}

/* 
 * Exploring dynamic parallelism in CUDA.  Namely, instead of staring
 * new kernels from CPU, we run new kernel for each _choice.
 *
 * @private
 */
__global__ void btDynamic(int32_t n, int32_t *choices, int32_t total, int index, int min, int max) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;   

    if (total > 0) {
        // printf("0. tid=%d min=%d max=%d total=%d index=%d\n", tid, min, max, total, index);
        int *nchoices = (int*) malloc(sizeof(int) * total);
        for (int i = 0; i < total - 1; i++) {
            nchoices[i] = choices[i];
        }
        nchoices[total-1] = min + tid;
        __syncthreads();
        if (tid == 0) {
            free(choices);
        }
        choices = nchoices;
    }

    Node nodeHeap[POOL_SIZE];
    Env env(nodeHeap);

    BT bt(&env);
    for (int i = 0; i < n; ++i) {
        int op = _choiceDynamic(n, choices, &total, &index, 0, 1);
        int value = _choiceDynamic(n, choices, &total, &index, 0, n - 1);
        if (op == 0) {
            bt.add(value);
        } else {
            bt.remove(value);
        }
    }

    free(choices);
    _countIf(1);
}

/* @private */
__global__ void btSeqGPU(const int32_t bck_active, const int32_t n, const int32_t print_id) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= bck_active) {
        return;
    }
    // btDynamic<<<1, 1>>>(n, NULL, 0, 0, 0, 0);
    btDriver(n, print_id);
}

int main(int argc, char *argv[]) {
    return seqMain(argc, argv, (void (*)(...)) btSeqGPU, btSeqCPU);
}
