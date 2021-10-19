#ifndef RANGE_THREADS_H
#define RANGE_THREADS_H

#include <stdint.h>
#include <math.h>
#include "consts.h"
#include "errors.h"

#define INITIAL_THREADS 10000
#define MAX_CHOICES 20

#define STATUS_SIZE 2

#define INVALID_VALUE -1

#define INDEX(x,y) ((x) * STATUS_SIZE + (y))

typedef struct {
    // holds two values for each thread: number of threads in current
    // thread group, and the id of the first thread in that group
    int64_t* thread_status;
    int32_t launch_index;
    int64_t index_offset;
} ThreadState;

typedef struct {
    unsigned long long int *if_counter;
    unsigned long long int *total_if_counter;
    unsigned long long int num_choice_invocations;
    unsigned long long int num_thread_groups;
    // number of threads still running.
    unsigned long long int num_active_threads;
} ThreadStats;

extern ThreadStats *_bck_stats;
extern int32_t active_threads;

typedef struct {
    // 0 if range phase, 1 otherwise
    int8_t phase;
    int32_t running_threads;
    int64_t* estimated_threads;
} RangeState;

#define BACKTRACK(name, args, num_args)

/* @public */
#define EXPLORE(...) \
{ \
    starting_threads = min(INITIAL_THREADS, MAX_THREADS_PER_BLOCK); \
    starting_blocks = (INITIAL_THREADS > MAX_THREADS_PER_BLOCK) ? \
     (INITIAL_THREADS / starting_threads) + 1 : INITIAL_THREADS / starting_threads; \
    printf("Running with %d blocks and %d threads\n", starting_blocks, starting_threads); \
    active_threads = INITIAL_THREADS; \
    RangeState* range_state = _rangeStateInit(starting_blocks, starting_threads); \
    float range_time; \
    cudaEvent_t range_start, range_stop; \
    CUDA_CHECK_RETURN(cudaEventCreate(&range_start)); \
    CUDA_CHECK_RETURN(cudaEventCreate(&range_stop)); \
    CUDA_CHECK_RETURN(cudaEventRecord(range_start)); \
    __VA_ARGS__; \
    CUDA_CHECK_RETURN(cudaEventRecord(range_stop)); \
    CUDA_CHECK_RETURN(cudaEventSynchronize(range_stop)); \
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&range_time, range_start, range_stop)); \
    printf("Ranging time: %.2lf\n", range_time);\
    range_state->phase = 1; \
    int64_t total_threads = calculateTotalThreads(range_state); \
    int32_t kernel_launches = calculateKernelLaunches(ALL_THREADS, total_threads); \
    int32_t max_threads = min(ALL_THREADS, MAX_THREADS_PER_BLOCK); \
    int32_t max_blocks = ceil((ALL_THREADS + 0.0) / (MAX_THREADS_PER_BLOCK + 0.0)); \
    int64_t offset = ALL_THREADS; \
    printf("total threads %ld kernel launches %d\n", total_threads, kernel_launches); \
    ThreadStats *thread_stats = _threadStatsInit(max_threads * max_blocks); \
    ThreadState *thread_state = _threadStateInit(ALL_THREADS); \
    int64_t remaining_threads = total_threads; \
    while (thread_state->launch_index < kernel_launches) { \
        printf("max threads %d max blocks %d\n", max_threads, max_blocks); \
        int64_t current_threads = min(remaining_threads, (int64_t) ALL_THREADS); \
        starting_threads = (int32_t) min(current_threads, (int64_t) max_threads); \
        starting_blocks = ceil((current_threads + 0.0) / (MAX_THREADS_PER_BLOCK + 0.0)); \
        int32_t reset_threads = min(starting_threads * starting_blocks, ALL_THREADS); /* so we don't reset the whole state for smaller sizes */ \
        active_threads = current_threads; \
        _threadStateReset(thread_state, reset_threads, total_threads, (int64_t) thread_state->launch_index * offset); \
        printf("Kernel launch %d, threads %d blocks %d\n", thread_state->launch_index, starting_threads, starting_blocks); \
        cudaEvent_t temp_start, temp_stop; \
        CUDA_CHECK_RETURN(cudaEventCreate(&temp_start)); \
        CUDA_CHECK_RETURN(cudaEventCreate(&temp_stop)); \
        CUDA_CHECK_RETURN(cudaEventRecord(temp_start)); \
        __VA_ARGS__; \
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
        CUDA_CHECK_RETURN(cudaEventRecord(temp_stop)); \
        CUDA_CHECK_RETURN(cudaEventSynchronize(temp_stop)); \
        CUDA_CHECK_RETURN(cudaEventElapsedTime(&range_time, temp_start, temp_stop)); \
        printf("Kernel launch %d, time: %.2lf\n", thread_state->launch_index, range_time);\
        printf("Total threads before %ld\n", remaining_threads); \
        remaining_threads = remaining_threads - current_threads; \
        printf("Total threads after %ld\n", remaining_threads); \
        thread_state->launch_index++; \
    } \
    _bck_stats = thread_stats; \
    _threadStatsPrint(thread_stats); \
}

/* @private */
RangeState* _rangeStateInit(int32_t, int32_t);

/* @private */
int64_t calculateTotalThreads(RangeState*);

/* @private */
int32_t calculateKernelLaunches(int32_t, int64_t);

/* @private */
void _bckAlg(void (*k)(void*), void*);

/* @private */
ThreadState* _threadStateInit(int64_t);

/* @private */
void _threadStateReset(ThreadState*, int32_t, int64_t, int64_t);

/* @private */
ThreadStats* _threadStatsInit(int32_t);

/* @private */
void _threadStatsPrint(ThreadStats*);

#endif