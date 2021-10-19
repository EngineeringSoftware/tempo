
#ifndef BACKTRACK_H
#define BACKTRACK_H

#include <stdint.h>
#include "consts.h"
#include "errors.h"

// Max size of overflow worklist
#define MAX_OVERFLOW_WL 8192 * 5 * 1000

// Total, index, 30 choices (per thread/task)
// #define MAX_CHOICES 18
#define MAX_CHOICES 25

// Convert 2D index to 1D index (used for output worklist)
#define INDEX(x,y) ((x) * MAX_CHOICES + (y))
#define INDEX_PARTIAL(x) ((x) * MAX_CHOICES)
#define INDEX_EVAL(val, y) ((val) + (y))

// typedef int16_t wl_T;
typedef int8_t wl_T;

// Number of active threads in each call
extern int32_t active_threads;

// Overflow worklist
extern wl_T *overflow_wl;

// Overflow row
extern unsigned int overflow_row;

// State of backtracking algorithm
typedef struct {
    // input worklist
    wl_T *in;
    // output worklist
    wl_T *out;
    // index in the output worklist for the next "row"
    unsigned int *row;
} BckState;

// Stats of backtracking algorithm
typedef struct {
    /* total number of tasks */
    unsigned int num_tasks_created;
    /* total number of choice invocations */
    unsigned int num_choice_invocations;
    /* number of threads on the last level */
    unsigned int num_last_level;
    /* number of kernel invocations */
    unsigned int num_kernel_invocations;
    /* counter available on device via API */
    unsigned long long int *if_counter;
    /* total number of times count_if is called */
    unsigned long long int *total_if_counter;
    /* max number of threads across all iterations */
    unsigned int max_num_of_threads;
} BckStats;

extern BckStats *_bck_stats;

/*
 * Explores all choices of the given named kernel; arguments are
 * passed to the kernel.
 *
 * @public
 */
#define BACKTRACK(name, args, num_args) \
 { \
   printf("Running example " #name " ...\n"); \
   _bckAlg((name), (args), (num_args)); \
   CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
 }

/*
 * See the backtracking loop for explanation of code below; this code
 * should be consistent with code in the loop.
 *
 * @public
 */
#define EXPLORE(...) \
 { \
   printf("Running example ...\n"); \
   BckState *state = _bckStateInit(); \
   BckStats *stats = _bckStatsInit(); \
   unsigned int input_count; \
   starting_blocks = 1; \
   starting_threads = 1; \
   active_threads = 1; \
   do { \
       _bckLoopPrefix(state, stats); \
       __VA_ARGS__; \
       CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
       _bckLoopSuffix(state, stats); \
       input_count = *(state->row); \
       stats->num_tasks_created += input_count; \
       stats->num_kernel_invocations++; \
       starting_threads = (input_count < MAX_THREADS_PER_BLOCK) ? input_count : MAX_THREADS_PER_BLOCK; \
       starting_blocks = (input_count + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK; \
       active_threads = input_count; \
       stats->max_num_of_threads = max(stats->max_num_of_threads, input_count); \
   } while (input_count > 0); \
   _bckPrintStats(stats, state);  \
   _bckStateDestroy(state); \
   _bck_stats = stats; \
   CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
 }
// we do not destroy stats above to be able to give them to users
// _bckStatsDestroy(stats);                  \

/*
 * Returns the number of threads in the last run of the backtracking loop.
 */
uint32_t bckGetNumOfThreadsInLastRun(void);

/* ---------------------------------------- */

/* Function below are here so that we can use them in macros above */

/*
 * Generic function for starting a kernel on GPU that executes a
 * backtracing algorithm. The second argument is "arguments" to be
 * passed to the kernel.
 *
 * @private
 */
void _bckAlg(void (*k)(...), void*, int);

/* @private */
BckState* _bckStateInit(void);

/* @private */
void _bckStateDestroy(BckState*);

/* @private */
BckStats* _bckStatsInit(void);

/* @private */
void _bckStatsDestroy(BckStats*);

/* @private */
void _bckLoopPrefix(BckState*, BckStats*);

/* @private */
void _bckLoopSuffix(BckState*, BckStats*);

/* @private */
void _bckPrintStats(BckStats*, BckState*);

/* @private */
void _copyOutToInWL(BckState*, unsigned int);

/* @private */
__global__ void _copyWorklistKernel(wl_T*, wl_T*, unsigned int);

/* @private */
void _moveToOverflowWL(BckState*, unsigned int);

/* @private */
int _moveFromOverflowWL(BckState*, unsigned int);

#endif
