#ifndef ALL_THREADS_CPU_H
#define ALL_THREADS_CPU_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "omp.h"

#define ESTIMATION_THREADS 10
#define MAX_THREADS 12
#define INVALID_VALUE -1
#define STATUS_SIZE 2
#define INDEX(x,y) ((x) * STATUS_SIZE + (y))

typedef struct {
    // 0 if range phase, 1 otherwise
    int8_t phase;
    int32_t running_threads;
    int64_t* estimated_threads;
} EstimationState;

typedef struct {
    // holds two values for each thread: number of threads in current
    // thread group, and the id of the first thread in that group
    int64_t* thread_status;
    int64_t launch_index;
    int64_t index_offset;
} ThreadState;

typedef struct {
    unsigned long long int if_counter;
    unsigned long long int total_if_counter;
    unsigned long long int num_choice_invocations;
    unsigned long long int num_thread_groups;
} ThreadStats;

extern ThreadState* thread_state;
extern ThreadStats* thread_stats;

int64_t getNumThreads(int tid);

void setNumThreads(int tid, int64_t num_threads);

int64_t getThreadIndex(int tid);

void setThreadIndex(int tid, int64_t thread_index);

ThreadState* _threadStateInit(int64_t total_threads);

ThreadStats* _threadStatsInit(int64_t total_threads);

void _threadStatsPrint(ThreadStats* stats);

void _threadStateReset(ThreadState* state, int32_t state_size, int64_t total_threads, int64_t offset);

EstimationState* _estimationStateInit(int32_t threads);

int64_t calculateKernelLaunches(int64_t max_threads_per_launch, int64_t total_threads);

int32_t estimationChoice(int32_t min, int32_t max);

int32_t _choice(int32_t min, int32_t max);

void _ignoreIf(int32_t condition);

void _countIf(int32_t conditions);

#endif
