#include <stdio.h>
#include <math.h>

#include "allthreads.h"
#include "errors.h"

ThreadStats *_bck_stats = NULL;
__device__ ThreadState *d_thread_state = NULL;
__device__ ThreadStats *d_thread_stats = NULL;

__device__ RangeState *d_range_state = NULL;

int32_t starting_threads = min(INITIAL_THREADS, MAX_THREADS_PER_BLOCK);
int32_t starting_blocks = (INITIAL_THREADS > MAX_THREADS_PER_BLOCK) ? \
 (INITIAL_THREADS / starting_threads) + 1 : INITIAL_THREADS / starting_threads; \
int32_t active_threads = starting_blocks * starting_threads;

__device__ void _ignoreIf(int32_t condition) {
    if (d_range_state->phase == 0) {
        return;
    }

    if (condition) {
        #ifndef DISABLE_CHOICE_DEBUG
        int64_t idx = (int64_t) threadIdx.x + (int64_t) blockDim.x * (int64_t) blockIdx.x;
        if (d_thread_state->thread_status[INDEX(idx, 1)] == 0) {
            atomicAdd(d_thread_stats->total_if_counter, 1);
        }
        #endif
        asm("exit;");
    }
}

__device__ void _countIf(int32_t condition) {
    #ifndef DISABLE_CHOICE_DEBUG
    if (d_range_state->phase == 0) {
        return;
    }

    int64_t idx = (int64_t) threadIdx.x + (int64_t) blockDim.x * (int64_t) blockIdx.x;
    // make sure that only the first thread of each thread group
    // increments the if_counter
    if (condition) {
        if (d_thread_state->thread_status[INDEX(idx, 1)] == 0) {
            atomicAdd(d_thread_stats->total_if_counter, 1);
            atomicAdd(d_thread_stats->if_counter, 1);
        }
    }
    #endif
}

__device__ int32_t _choice(int32_t min, int32_t max) {
    int64_t idx = (int64_t) threadIdx.x + (int64_t) blockDim.x * (int64_t) blockIdx.x;

    if (d_range_state->phase == 0) {
        if (idx >= INITIAL_THREADS) {
            return -1;
        }

        int64_t current_width = ((int64_t) max - (int64_t) min + 1);
        d_range_state->estimated_threads[idx] *= current_width;
        if (idx == 0) {
            return min;
        } else if (idx == INITIAL_THREADS - 1) {
            return max;
        } else {
            return min + (idx % (max - min + 1));
        }
    } else {
        if (min == max) {
            return min;
        } else if (min > max) {
            // printf("ERROR: min %d is less than max %d\n", min, max);
            asm("exit;");
        }

        const int64_t current_thread_count = d_thread_state->thread_status[INDEX(idx, 0)];
        const int64_t thread_index = d_thread_state->thread_status[INDEX(idx, 1)];
        const int64_t num_choices = (int64_t) max - (int64_t) min + (int64_t) 1;

        #ifndef DISABLE_CHOICE_DEBUG
        if (thread_index == 0) {
            atomicAdd(&d_thread_stats->num_choice_invocations, 1);
            atomicAdd(&d_thread_stats->num_thread_groups, num_choices - 1);
        }
        
        if (current_thread_count / num_choices == 0 && thread_index == 0) {
            printf("ERROR: no more threads available\n");
            asm("trap;");
        }
        #endif

        const int64_t new_thread_count = current_thread_count / num_choices;
        const int64_t new_thread_index = thread_index % new_thread_count;

        d_thread_state->thread_status[INDEX(idx, 0)] = new_thread_count;
        d_thread_state->thread_status[INDEX(idx, 1)] = new_thread_index;

        const int32_t choice_value = min + (thread_index / new_thread_count);

        if (choice_value > max) {
            d_thread_state->thread_status[INDEX(idx, 0)] = INVALID_VALUE;
            d_thread_state->thread_status[INDEX(idx, 1)] = INVALID_VALUE;
            asm("exit;");
        }

        return choice_value;
    }
}

void explore(void (*k)(...), void *args, int num_args) {
    int* cast_args;
    if (args != NULL)
        cast_args = (int*)args;

    int starting_threads = min(INITIAL_THREADS, MAX_THREADS_PER_BLOCK);
    int starting_blocks = (INITIAL_THREADS > MAX_THREADS_PER_BLOCK) ?
        (INITIAL_THREADS / starting_threads) + 1 : INITIAL_THREADS / starting_threads; \

    printf("Running with %d blocks and %d threads\n", starting_blocks, starting_threads);
    int active_threads = INITIAL_THREADS;

    RangeState* range_state = _rangeStateInit(starting_blocks, starting_threads);

    float range_time;
    cudaEvent_t range_start, range_stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&range_start));
    CUDA_CHECK_RETURN(cudaEventCreate(&range_stop));
    CUDA_CHECK_RETURN(cudaEventRecord(range_start));

    if (num_args == 0)
        k<<<starting_blocks, starting_threads>>>(active_threads);
    else if (num_args == 1)
        k<<<starting_blocks, starting_threads>>>(active_threads, cast_args[0]);
    else
        k<<<starting_blocks, starting_threads>>>(active_threads, cast_args[0], cast_args[1]);

    CUDA_CHECK_RETURN(cudaEventRecord(range_stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(range_stop));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&range_time, range_start, range_stop));
    printf("Ranging time: %.2lf\n", range_time);

    range_state->phase = 1;
    int64_t total_threads = calculateTotalThreads(range_state);
    int32_t kernel_launches = calculateKernelLaunches(ALL_THREADS, total_threads);
    int32_t max_threads = min(ALL_THREADS, MAX_THREADS_PER_BLOCK);
    int32_t max_blocks = ceil((ALL_THREADS + 0.0) / (MAX_THREADS_PER_BLOCK + 0.0));
    int64_t offset = ALL_THREADS;
    printf("total threads %ld kernel launches %d\n", total_threads, kernel_launches);

    ThreadStats *thread_stats = _threadStatsInit(max_threads * max_blocks);
    ThreadState *thread_state = _threadStateInit(ALL_THREADS);
    int64_t remaining_threads = total_threads;

    while (thread_state->launch_index < kernel_launches) {
        printf("max threads %d max blocks %d\n", max_threads, max_blocks);
        int64_t current_threads = min(remaining_threads, (int64_t) ALL_THREADS);
        starting_threads = (int32_t) min(current_threads, (int64_t) max_threads);
        starting_blocks = ceil((current_threads + 0.0) / (MAX_THREADS_PER_BLOCK + 0.0));

        // so we don't reset the whole state for smaller sizes
        int32_t reset_threads = min(starting_threads * starting_blocks, ALL_THREADS);
        active_threads = current_threads;

        _threadStateReset(thread_state, reset_threads, total_threads, (int64_t) thread_state->launch_index * offset);
        printf("Kernel launch %d, threads %d blocks %d\n", thread_state->launch_index, starting_threads, starting_blocks);

        cudaEvent_t temp_start, temp_stop;
        CUDA_CHECK_RETURN(cudaEventCreate(&temp_start));
        CUDA_CHECK_RETURN(cudaEventCreate(&temp_stop));
        CUDA_CHECK_RETURN(cudaEventRecord(temp_start));

        if (num_args == 0)
            k<<<starting_blocks, starting_threads>>>(active_threads);
        else if (num_args == 1)
            k<<<starting_blocks, starting_threads>>>(active_threads, cast_args[0]);
        else
            k<<<starting_blocks, starting_threads>>>(active_threads, cast_args[0], cast_args[1]);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        CUDA_CHECK_RETURN(cudaEventRecord(temp_stop));
        CUDA_CHECK_RETURN(cudaEventSynchronize(temp_stop));
        CUDA_CHECK_RETURN(cudaEventElapsedTime(&range_time, temp_start, temp_stop));

        printf("Kernel launch %d, time: %.2lf\n", thread_state->launch_index, range_time);
        printf("Total threads before %ld\n", remaining_threads);
        remaining_threads = remaining_threads - current_threads;
        printf("Total threads after %ld\n", remaining_threads);
        thread_state->launch_index++;
    }
    _bck_stats = thread_stats;
    _threadStatsPrint(thread_stats);
}

ThreadState* _threadStateInit(int64_t total_threads) {
    ThreadState *state;
    
    int64_t thread_status_size = total_threads * STATUS_SIZE;
    CUDA_MALLOC_MANAGED(&state, sizeof(ThreadState));
    CUDA_MALLOC_MANAGED(&state->thread_status, sizeof(int64_t) * thread_status_size);
    CUDA_MEMCPY_TO_SYMBOL(d_thread_state, &state, sizeof(ThreadState*));

    state->launch_index = 0;

    return state;
}

ThreadStats* _threadStatsInit(int32_t total_threads) {
    ThreadStats *stats;

    CUDA_MALLOC_MANAGED(&stats, sizeof(ThreadStats));
    CUDA_MALLOC_MANAGED(&stats->if_counter, sizeof(unsigned long long int));
    CUDA_MALLOC_MANAGED(&stats->total_if_counter, sizeof(unsigned long long int));
    CUDA_MEMCPY_TO_SYMBOL(d_thread_stats, &stats, sizeof(ThreadStats*));

    *(stats->if_counter) = 0;
    *(stats->total_if_counter) = 0;
    stats->num_choice_invocations = 0;
    stats->num_thread_groups = 1;
    active_threads = total_threads;
    stats->num_active_threads = active_threads;

    return stats;
}

RangeState* _rangeStateInit(int32_t blocks, int32_t threads) {
    RangeState* state;

    int32_t estimated_threads_size = blocks * threads;
    CUDA_MALLOC_MANAGED(&state, sizeof(RangeState));
    CUDA_MALLOC_MANAGED(&state->estimated_threads, sizeof(int64_t) * estimated_threads_size);
    CUDA_MEMCPY_TO_SYMBOL(d_range_state, &state, sizeof(RangeState*));

    state->phase = 0;
    state->running_threads = estimated_threads_size;
    for (int i = 0; i < estimated_threads_size; i++) {
        state->estimated_threads[i] = 1;
    }

    return state;
}

void _threadStateReset(ThreadState* state, int32_t state_size, int64_t total_threads, int64_t offset) {
    for (int32_t i = 0; i < state_size; i++) {
        state->thread_status[INDEX(i, 0)] = total_threads;
        state->thread_status[INDEX(i, 1)] = offset + i;
    }
    state->index_offset = offset;
}

int64_t calculateTotalThreads(RangeState* state) {
    int64_t total_threads = 1;
    for (int i = 0; i < state->running_threads; i++) {
        total_threads = max(total_threads, state->estimated_threads[i]);
    }
    return total_threads;
}

int32_t calculateKernelLaunches(int32_t max_threads_per_launch, int64_t total_threads) {
    return ceil(((total_threads + 0.0) / (max_threads_per_launch + 0.0)));
}

void _threadStatsPrint(ThreadStats *stats) {
    #ifdef DEBUG
    printf("# of tasks created: %llu\n", stats->num_thread_groups);
    printf("# of choice invocations: %llu\n", stats->num_choice_invocations);
    printf("# of threads on the last level: %llu\n", stats->num_active_threads);
    printf("# if_counter: %llu\n", *(stats->if_counter));
    printf("# total_threads: %llu\n", *(stats->total_if_counter));
    printf("# max_threads: %llu\n", stats->num_active_threads);
    #endif
}
