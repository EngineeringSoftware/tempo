
#include <stdio.h>
#include <time.h>

#include "errors.h"
#include "exactthreads.h"

__device__ BckState *d_bck_state = NULL;
__device__ BckStats *d_bck_stats = NULL;

int32_t starting_blocks = 1;
int32_t starting_threads = 1;
int32_t active_threads = 1;
BckStats *_bck_stats = NULL;
wl_T *overflow_wl = NULL;
unsigned int overflow_row = 0;
double suffix_time_2 = 0;
double suffix_time_3 = 0;

/*
 * Implementation of this function populates 'out' worklist with new
 * choices, or return old choices.
 */
__device__ int _choice(const int min, const int max) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idx_partial = INDEX_PARTIAL(idx);
    #ifndef DISABLE_CHOICE_DEBUG
    if (idx >= MAX_IN_WL) {
        printf("ERROR: %d out of the worklists\n", idx);
        asm("exit;");
    }
    #endif

    // if min and max are same, we do not want choice.
    if (min == max) {
        return min;
    }

    // if we are running only a single path, then we always use min value.
    if (d_bck_state == NULL) {
        return min;
    }
    // increment number of choice invocations

    #ifndef DISABLE_CHOICE_DEBUG
    atomicAdd(&d_bck_stats->num_choice_invocations, 1);
    #endif

    wl_T *const in = d_bck_state->in;
    wl_T *const out = d_bck_state->out;

    const wl_T total = in[INDEX_EVAL(idx_partial, 0)];
    const wl_T index = in[INDEX_EVAL(idx_partial, 1)];

    if (index >= (total + 2)) { 
        // Since total is at index 0, and index is at index 1,
        // index will start at 2 by default. This branch is taken
        // when index is pointing to a new _choice call. 

        const int num_choices = max - min + 1;
        unsigned int row = atomicAdd(d_bck_state->row, num_choices);

        // create new entries in the 'out' worklists
        for (int i = min; i <= max; i++) {
            #ifndef DISABLE_CHOICE_DEBUG
            if (INDEX(row, total + 1) > (MAX_OUT_WL * MAX_CHOICES)) {
                printf("ERROR: Index out of bounds (%d/%d)\n",
                       INDEX(row, total + 1), MAX_OUT_WL * MAX_CHOICES);
                asm("trap;");
            }
            #endif

            // copy the current (idx) row to the new row
            out[INDEX(row, 0)] = total + 1;
            out[INDEX(row, 1)] = 2;

            int j;
            for (j = 2; j < total + 2; j++) {
                // copy the previously selected values of old _choice calls 
                out[INDEX(row, j)] = in[INDEX_EVAL(idx_partial, j)];
            }
            out[INDEX(row, j)] = i;
            row++;
        }
        asm("exit;");
        return 0;
    } else {
        // this is an old choice
        // increment index
        in[INDEX_EVAL(idx_partial, 1)] = index + 1;
        // printf("[choice=%d]\n", in[INDEX(idx, index)]);
        return in[INDEX_EVAL(idx_partial, index)];
    }
}

__device__ void _ignoreIf(const int32_t condition) {
    if (condition) {
        asm("exit;");
    }
}

__device__ void _countIf(const int32_t condition) {
    #ifndef DISABLE_CHOICE_DEBUG
    atomicAdd(d_bck_stats->total_if_counter, 1);
    if (condition) {
        atomicAdd(d_bck_stats->if_counter, 1);
    }
    #endif
}

uint32_t bckGetNumOfThreadsInLastRun(void) {
    return _bck_stats->num_last_level;
}

// PRIVATE

// initialize global device pointer d_bck_state 
// use unified memory to allocate worklists
BckState* _bckStateInit(void) {
    BckState *state;

    CUDA_MALLOC_MANAGED(&state, sizeof(BckState));
    CUDA_MALLOC(&state->in, sizeof(wl_T) * MAX_CHOICES * MAX_IN_WL);
    CUDA_MALLOC(&state->out, sizeof(wl_T) * MAX_CHOICES * MAX_OUT_WL);
    CUDA_MALLOC(&state->row, sizeof(unsigned int));
    CUDA_MEMCPY_TO_SYMBOL(d_bck_state, &state, sizeof(BckState*));

    wl_T* in_wl = (wl_T*) malloc(MAX_IN_WL * sizeof(wl_T) * MAX_CHOICES);

    overflow_wl = (wl_T*) malloc(MAX_OVERFLOW_WL * sizeof(wl_T) * MAX_CHOICES);
    overflow_wl = (wl_T*) malloc(MAX_OVERFLOW_WL * sizeof(wl_T) * MAX_CHOICES);
    overflow_row = 0;

    // initialize in worklist
    for (int i = 0; i < MAX_IN_WL; ++i) {
        in_wl[INDEX(i, 0)] = 0;
        in_wl[INDEX(i, 1)] = 2;
    }

    CUDA_CHECK_RETURN(cudaMemcpy(state->in, in_wl, MAX_IN_WL * sizeof(wl_T) * MAX_CHOICES, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    return state;
}

void _bckStateDestroy(BckState *state) {
    // free memory
    CUDA_FREE(state->in);
    CUDA_FREE(state->out);
    CUDA_FREE(state->row);
    CUDA_FREE(state);
    // set device pointer to NULL
    state = NULL;
    CUDA_MEMCPY_TO_SYMBOL(d_bck_state, &state, sizeof(BckState*));
}

BckStats* _bckStatsInit(void) {
    BckStats *stats;

    CUDA_MALLOC_MANAGED(&stats, sizeof(BckStats));
    CUDA_MALLOC_MANAGED(&stats->if_counter, sizeof(unsigned long long int));
    CUDA_MALLOC_MANAGED(&stats->total_if_counter, sizeof(unsigned long long int));
    CUDA_MEMCPY_TO_SYMBOL(d_bck_stats, &stats, sizeof(BckStats*));

    stats->num_tasks_created = 0;
    stats->num_choice_invocations = 0;
    stats->num_last_level = 0;
    stats->num_kernel_invocations = 0;
    *(stats->if_counter) = 0;
    *(stats->total_if_counter) = 0;
    stats->max_num_of_threads = 0;

    return stats;
}

void _bckStatsDestroy(BckStats *stats) {
    // free memory
    CUDA_FREE(stats->if_counter);
    CUDA_FREE(stats->total_if_counter);
    CUDA_FREE(stats);
    // set device pointer to NULL
    stats = NULL;
    CUDA_MEMCPY_TO_SYMBOL(d_bck_stats, &stats, sizeof(BckStats*));
}

void _bckLoopPrefix(BckState *const state, BckStats *const stats) {
    // save number of threads on the last level
    unsigned int row = 0;
    CUDA_CHECK_RETURN(cudaMemcpy(&row, state->row, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    stats->num_last_level = row;

    // reset size of out
    row = 0;
    CUDA_CHECK_RETURN(cudaMemcpy(state->row, &row, sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void _bckLoopSuffix(BckState *const state, BckStats *const stats) {
    // move from in to out
    clock_t start = clock();

    unsigned int row = 0;
    CUDA_CHECK_RETURN(cudaMemcpy(&row, state->row, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned int next_in_size = min(row, MAX_IN_WL);
    _copyOutToInWL(state, next_in_size);
    suffix_time_2 += (double) (clock() - start) / CLOCKS_PER_SEC;

    start = clock();
    // move remaining items from out to overflow
    _moveToOverflowWL(state, next_in_size);
    suffix_time_3 += (double) (clock() - start) / CLOCKS_PER_SEC;

    // move to remaining space left inside in
    unsigned int items_moved = _moveFromOverflowWL(state, next_in_size);

    row = next_in_size + items_moved;
    CUDA_CHECK_RETURN(cudaMemcpy(state->row, &row, sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void explore(void (*k)(...), void *const args, const int num_args) {
    int *cast_args;
    if (args != NULL) {
        cast_args = (int*) args;
    }

    BckState *const state = _bckStateInit();
    BckStats *const stats = _bckStatsInit();

    unsigned int thread_count = 1;
    unsigned int block_count = 1;
    unsigned int input_count;
    active_threads = 1;

    double prefix_time = 0.0;
    double kernel_time = 0.0;
    double suffix_time = 0.0;

    do {
        clock_t start = clock();
        _bckLoopPrefix(state, stats);
        prefix_time += (double) (clock() - start) / CLOCKS_PER_SEC;

        start = clock();

        if (num_args == 0) {
            k<<<block_count, thread_count>>>(active_threads);
        } else if (num_args == 1) {
            k<<<block_count, thread_count>>>(active_threads, cast_args[0]);
        } else {
            k<<<block_count, thread_count>>>(active_threads, cast_args[0], cast_args[1]);
        }

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        kernel_time += (double) (clock() - start) / CLOCKS_PER_SEC;

        start = clock();
        _bckLoopSuffix(state, stats);

        // number of inputs for the next iteration
        cudaMemcpy(&input_count, state->row, sizeof(int), cudaMemcpyDeviceToHost);
        active_threads = input_count;

        // increment number of tasks
        stats->num_tasks_created += input_count;
        stats->num_kernel_invocations++;
        stats->max_num_of_threads = max(stats->max_num_of_threads, input_count);

        // calculate thread and block counts
        thread_count = (input_count < MAX_THREADS_PER_BLOCK) ? input_count : MAX_THREADS_PER_BLOCK;
        block_count = (input_count + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
        suffix_time += (double) (clock() - start) / CLOCKS_PER_SEC;
    } while (input_count > 0);

    _bckPrintStats(stats, state);
    _bckStateDestroy(state);
    _bck_stats = stats;
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    double total_time = prefix_time + kernel_time + suffix_time;
    printf("prefix %f kernel %f suffix %f total %f suffix_2 %f suffix_3 %f\n", prefix_time, kernel_time, suffix_time, total_time, suffix_time_2, suffix_time_3);
}

void _bckPrintStats(BckStats *stats, BckState *state) {
    #ifdef DEBUG
    printf("# of tasks created     : %u\n", stats->num_tasks_created);
    printf("# of choice invocations: %u\n", stats->num_choice_invocations);
    printf("# of threads on the last level: %u\n", stats->num_last_level);
    printf("# of kernel invocations: %u\n", stats->num_kernel_invocations);
    printf("# if_counter: %llu\n", *(stats->if_counter));
    printf("# total_threads: %llu\n", *(stats->total_if_counter));
    printf("# max_threads: %u\n", stats->max_num_of_threads);
    #endif
}

// Initialize in, out worklists and call kernel with arguments
// for all possible choice paths
void _bckAlg(void (*k)(...), void *args, int num_args) {
    BckState *state = _bckStateInit();
    BckStats *stats = _bckStatsInit();

    explore(k, args, num_args);
    _bckPrintStats(stats, state);

    _bckStateDestroy(state);
    _bckStatsDestroy(stats);
}

void _copyOutToInWL(BckState *state, unsigned int next_in_size) {
    if (next_in_size == 0) {
        return;
    }
    next_in_size = next_in_size * MAX_CHOICES;

    int thread_count = min(next_in_size, 1024);
    int block_count = (next_in_size + 1023) / 1024;
    _copyWorklistKernel<<<block_count, thread_count>>>(state->out, state->in, next_in_size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

__global__ void _copyWorklistKernel(
    wl_T* source,
    wl_T* destination,
    unsigned int size)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) {
        return;
    }

    destination[idx] = source[idx];
}

void _moveToOverflowWL(BckState* state, unsigned int next_in_size) {
    unsigned int row;
    cudaMemcpy(&row, state->row, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int remaining_out_items = row - next_in_size;
    if (remaining_out_items <= 0) {
        return;
    }
    if (overflow_row + remaining_out_items >= MAX_OVERFLOW_WL) {
        printf("ERROR: overflow worklist is full\n");
        exit(1);
        return;
    }

    unsigned int dst_offset = overflow_row * MAX_CHOICES;
    unsigned int src_offset = next_in_size  * MAX_CHOICES;

    int64_t size = (int64_t) remaining_out_items * sizeof(wl_T) * MAX_CHOICES;

    CUDA_CHECK_RETURN(cudaMemcpy(&overflow_wl[dst_offset], &state->out[src_offset], size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    overflow_row = overflow_row + remaining_out_items;
}

// returns the number of items moved
int _moveFromOverflowWL(BckState* state, unsigned int next_in_size) {
    if (overflow_row == 0) {
        return 0;
    }

    unsigned int remaining_in_space = MAX_IN_WL - next_in_size;
    if (remaining_in_space == 0) {
        return 0;
    }

    // calculate how many items to move to in
    unsigned int move_from_overflow = min(overflow_row, remaining_in_space);
    // calculate the offset from where copying should start
    unsigned int overflow_offset = overflow_row - move_from_overflow;

    unsigned int dst_offset = next_in_size * MAX_CHOICES;
    unsigned int src_offset = overflow_offset  * MAX_CHOICES;

    CUDA_CHECK_RETURN(cudaMemcpy(&state->in[dst_offset], &overflow_wl[src_offset], sizeof(wl_T) * move_from_overflow * MAX_CHOICES, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    overflow_row = overflow_row - move_from_overflow;

    return move_from_overflow; 
}
