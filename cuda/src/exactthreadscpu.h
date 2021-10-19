#ifndef EXACT_THREADS_CPU
#define EXACT_THREADS_CPU

#include <stdint.h>
#include "omp.h"

#define MAX_THREADS 12
#define MAX_CHOICES 32
#define MAX_WL_SIZE 8192 * 5 * 1000
#define INVALID_VALUE -1
#define INDEX(x,y) ((x) * MAX_CHOICES + (y))

typedef int16_t wl_T;

// State of backtracking algorithm
typedef struct {
    wl_T* worklist;
    wl_T* buffer;

    int64_t size;
} BckState;

// Stats of backtracking algorithm
typedef struct {
    /* total number of tasks */
    unsigned int num_tasks_created;
    /* total number of choice invocations */
    unsigned int num_choice_invocations;
    /* number of kernel invocations */
    unsigned int num_kernel_invocations;
    /* counter available on device via API */
    unsigned long long int if_counter;
    /* total number of times count_if is called */
    unsigned long long int total_if_counter;
} BckStats;

/* @private */
BckState* _bckStateInit(void);

/* @private */
BckStats* _bckStatsInit(void);

/* @private */
void _bckPrintStats(BckStats* stats);

/* @private */
void cancelThread(int tid);

/* @private */
int8_t isCancelledThread(int tid);

int32_t _choice(int32_t min, int32_t max);

void _ignoreIf(int32_t condition);

void _countIf(int32_t conditions);

#endif