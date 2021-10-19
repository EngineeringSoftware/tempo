
#ifndef ALL_THREADS_H
#define ALL_THREADS_H

#include <stdint.h>
#include "consts.h"
#include "errors.h"

// Number of elements needed to keep track of the state of each thread
// Element 0 is the current thread count, and element 1 is the id of
// the first thread in that group of threads
#define STATUS_SIZE 2

#define INVALID_VALUE -1

#define INDEX(x,y) ((x) * STATUS_SIZE + (y))

typedef struct {
    // holds two values for each thread: number of threads in current
    // thread group, and the id of the first thread in that group
    int32_t* thread_status;
} ThreadState;

typedef struct {
    int32_t *if_counter;
    int32_t *total_if_counter;
    int32_t num_choice_invocations;
    int32_t num_thread_groups;
    // number of threads still running.
    int32_t num_active_threads;
} ThreadStats;

extern ThreadStats *_bck_stats;

// Number of active threads in each call
// TODO: have all the threads that stop decrement this number
extern int32_t active_threads;

#define BACKTRACK(name, args) \
 { \
   printf("Running example " #name " ...\n"); \
   _bckAlg((name), (args)); \
   CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
 }

/* @public */
#define EXPLORE(...) \
{ \
    printf("Running example ...\n"); \
    starting_threads = min(ALL_THREADS, MAX_THREADS_PER_BLOCK); \
    starting_blocks = (ALL_THREADS / starting_threads) + 1; \
    active_threads = starting_blocks * starting_threads; \
    ThreadStats *thread_stats = _threadStatsInit(); \
    ThreadState *thread_state = _threadStateInit(starting_blocks, starting_threads); \
    __VA_ARGS__; \
    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
    _bck_stats = thread_stats; \
    _threadStatsPrint(thread_stats); \
}

/* @private */
void _bckAlg(void (*k)(void*), void*);

/* @private */
ThreadState* _threadStateInit(int32_t, int32_t);

/* @private */
ThreadStats* _threadStatsInit();

/* @private */
void _threadStatsPrint(ThreadStats*);

#endif
