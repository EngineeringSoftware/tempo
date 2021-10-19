#include <stdio.h>

#include "constthreads.h"
#include "errors.h"

ThreadStats *_bck_stats = NULL;
__device__ ThreadState *d_thread_state = NULL;
__device__ ThreadStats *d_thread_stats = NULL;

int32_t starting_threads = min(ALL_THREADS, MAX_THREADS_PER_BLOCK);
int32_t starting_blocks = ALL_THREADS / starting_threads;
int32_t active_threads = starting_blocks * starting_threads;

__device__ void _ignoreIf(int32_t condition) {
    if (condition) {
        asm("exit;");
    }
}

__device__ void _countIf(int32_t condition) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // make sure that only the first thread of each thread group
    // increments the if_counter

    if (d_thread_state->thread_status[INDEX(idx, 1)] == idx) {
        atomicAdd(d_thread_stats->total_if_counter, 1);
        if (condition) {
            atomicAdd(d_thread_stats->if_counter, 1);
        }
    }
}

__device__ int32_t _choice(int32_t min, int32_t max) {
    if (min == max) {
        return min;
    } else if (min > max) {
        printf("ERROR: min %d is less than max %d\n", min, max);
        asm("exit;");
    }

    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int32_t current_thread_count = d_thread_state->thread_status[INDEX(idx, 0)];
    int32_t first_thread_id = d_thread_state->thread_status[INDEX(idx, 1)];
    int32_t num_of_choices = max - min + 1;

    if (idx == first_thread_id) {
        atomicAdd(&d_thread_stats->num_choice_invocations, 1);
        atomicAdd(&d_thread_stats->num_thread_groups, num_of_choices - 1);
    }

    if (current_thread_count / num_of_choices == 0) {
        printf("ERROR: no more threads available\n");
        asm("trap;");
    }

    int32_t new_thread_count = current_thread_count / num_of_choices;
    for (int i = 0; i < num_of_choices; i++) {
        if (idx < first_thread_id + (new_thread_count * (i + 1))) {
            d_thread_state->thread_status[INDEX(idx, 0)] = new_thread_count;
            d_thread_state->thread_status[INDEX(idx, 1)] = first_thread_id + (new_thread_count * i);
            return min + i;
        }
    }

    d_thread_state->thread_status[INDEX(idx, 0)] = INVALID_VALUE;
    d_thread_state->thread_status[INDEX(idx, 1)] = INVALID_VALUE;
    asm("exit;");
    return INVALID_VALUE;
}

ThreadState* _threadStateInit(int32_t blocks, int32_t threads) {
    ThreadState *state;
    
    int64_t thread_status_size = (int64_t)blocks * (int64_t)threads * STATUS_SIZE;
    CUDA_MALLOC_MANAGED(&state, sizeof(ThreadState));
    CUDA_MALLOC_MANAGED(&state->thread_status, sizeof(int32_t) * thread_status_size);
    CUDA_MEMCPY_TO_SYMBOL(d_thread_state, &state, sizeof(ThreadState*));

    for (int i = 0; i < blocks * threads; i++) {
        state->thread_status[INDEX(i, 0)] = blocks * threads;
        state->thread_status[INDEX(i, 1)] = 0;
    }

    return state;
}

ThreadStats* _threadStatsInit() {
    ThreadStats *stats;

    CUDA_MALLOC_MANAGED(&stats, sizeof(ThreadStats));
    CUDA_MALLOC_MANAGED(&stats->if_counter, sizeof(int32_t));
    CUDA_MALLOC_MANAGED(&stats->total_if_counter, sizeof(unsigned int));
    CUDA_MEMCPY_TO_SYMBOL(d_thread_stats, &stats, sizeof(ThreadStats*));

    *(stats->if_counter) = 0;
    *(stats->total_if_counter) = 0;
    stats->num_choice_invocations = 0;
    stats->num_thread_groups = 1;
    stats->num_active_threads = active_threads;

    return stats;
}

void _threadStatsPrint(ThreadStats *stats) {
    #ifdef DEBUG
    printf("# of tasks created: %d\n", stats->num_thread_groups);
    printf("# of choice invocations: %d\n", stats->num_choice_invocations);
    printf("# of threads on the last level: %d\n", stats->num_active_threads);
    printf("# if_counter: %d\n", *(stats->if_counter));
    printf("# total_threads: %u\n", *(stats->total_if_counter));
    printf("# max_threads: %d\n", stats->num_active_threads);
    #endif
}

void _bckAlg(void (*k)(void*), void *args) {
    starting_threads = min(MAX_IN_WL, MAX_THREADS_PER_BLOCK);
    starting_blocks = MAX_IN_WL / starting_threads;
    active_threads = starting_blocks * starting_threads;

    ThreadStats *thread_stats = _threadStatsInit();
    ThreadState *thread_state = _threadStateInit(starting_blocks, starting_threads);

    k<<<starting_blocks, starting_threads>>>(args);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    _threadStatsPrint(thread_stats);
}