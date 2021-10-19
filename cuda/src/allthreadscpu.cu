#include <assert.h>

#include "allthreadscpu.h"

ThreadState* thread_state = NULL;
ThreadStats* thread_stats = NULL;
EstimationState* estimation_state = NULL;

int64_t getNumThreads(int tid) {
    return thread_state->thread_status[INDEX(tid, 0)];
}

void setNumThreads(int tid, int64_t num_threads) {
    thread_state->thread_status[INDEX(tid, 0)] = num_threads;
}

int64_t getThreadIndex(int tid) {
    return thread_state->thread_status[INDEX(tid, 1)];
}

void setThreadIndex(int tid, int64_t thread_index) {
    thread_state->thread_status[INDEX(tid, 1)] = thread_index;
}

EstimationState* _estimationStateInit(int32_t threads) {
    EstimationState* state = (EstimationState*) malloc(sizeof(EstimationState));
    state->estimated_threads = (int64_t*) malloc(sizeof(int64_t) * threads);

    state->phase = 0;
    state->running_threads = threads;
    for (int i = 0; i < threads; i++) {
        state->estimated_threads[i] = 1;
    }

    return state;
}

ThreadState* _threadStateInit(int64_t total_threads) {
    ThreadState* state = (ThreadState*) malloc(sizeof(ThreadState));

    int64_t thread_status_size = total_threads * STATUS_SIZE;
    state->thread_status = (int64_t*) malloc(sizeof(int64_t) * thread_status_size);

    state->launch_index = 0;
    state->index_offset = 0;

    for (int64_t i = 0; i < total_threads; i++) {
        state->thread_status[INDEX(i, 0)] = total_threads;
        state->thread_status[INDEX(i, 1)] = 0;
    }

    return state;
}

ThreadStats* _threadStatsInit(int64_t total_threads) {
    ThreadStats* stats = (ThreadStats*) malloc(sizeof(ThreadStats));
    stats->if_counter = 0;
    stats->total_if_counter = 0;
    stats->num_choice_invocations = 0;
    stats->num_thread_groups = 0;

    return stats;
}

void _threadStatsPrint(ThreadStats* stats) {
    #ifdef DEBUG
    printf("# of tasks created: %llu\n", stats->num_thread_groups);
    printf("# of choice invocations: %llu\n", stats->num_choice_invocations);
    printf("# if_counter: %llu\n", stats->if_counter);
    printf("# total_threads: %llu\n", stats->total_if_counter);
    #endif
}

int64_t calculateKernelLaunches(int64_t max_threads_per_launch, int64_t total_threads) {
    return ceil(((total_threads + 0.0) / (max_threads_per_launch + 0.0)));
}

void _threadStateReset(ThreadState* state, int32_t state_size, int64_t total_threads, int64_t offset) {
    for (int32_t i = 0; i < state_size; i++) {
        setNumThreads(i, total_threads);
        setThreadIndex(i, offset + i);
    }
    state->index_offset = offset;
}

void _ignoreIf(int32_t condition) {
    if (estimation_state->phase == 0) {
        return;
    }
    
    if (condition) {
        int tid = omp_get_thread_num();

        setNumThreads(tid, INVALID_VALUE);
        setThreadIndex(tid, INVALID_VALUE);
    }
}

void _countIf(int32_t condition) {
    #ifndef DISABLE_CHOICE_DEBUG
    if (estimation_state->phase == 0) {
        return;
    }

    int tid = omp_get_thread_num();

    if (getThreadIndex(tid) == INVALID_VALUE) {
        return;
    }

    if (condition) {
        if (getThreadIndex(tid) == 0) {
            #pragma omp atomic
            thread_stats->total_if_counter++;

            #pragma omp atomic
            thread_stats->if_counter++;
        }
    }
    #endif
}

int32_t estimationChoice(int32_t min, int32_t max) {
    int tid = omp_get_thread_num();

    int64_t current_width = ((int64_t) max - (int64_t) min + 1);
    estimation_state->estimated_threads[tid] *= current_width;
    if (tid == 0) {
        return min;
    } else if (tid == ESTIMATION_THREADS - 1) {
        return max;
    } else {
        return min + (tid % (max - min + 1));
    }
}

int32_t _choice(int32_t min, int32_t max) {
    if (estimation_state->phase == 0) {
        return estimationChoice(min, max);
    }

    if (min == max) {
        return min;
    } else if (min > max) {
        #ifndef DISABLE_CHOICE_DEBUG
        printf("ERROR: min %d should be less than max %d\n", min, max);
        #endif
        return INVALID_VALUE; 
    }

    const int tid = omp_get_thread_num();
    const int64_t current_thread_count = getNumThreads(tid);
    const int64_t thread_index = getThreadIndex(tid);

    if (current_thread_count == INVALID_VALUE || current_thread_count == INVALID_VALUE) {
        return INVALID_VALUE;
    }

    const int64_t num_choices = (int64_t) max - (int64_t) min + (int64_t) 1;

    #ifndef DISABLE_CHOICE_DEBUG
    if (thread_index == 0) {
        #pragma omp atomic
        thread_stats->num_choice_invocations++;

        #pragma omp atomic
        thread_stats->num_thread_groups++;
    }
    
    if (current_thread_count / num_choices == 0 && thread_index == 0) {
        printf("ERROR: no more threads available\n");
        return INVALID_VALUE;
    }
    #endif

    const int64_t new_thread_count = current_thread_count / num_choices;
    const int64_t new_thread_index = thread_index % new_thread_count;

    setNumThreads(tid, new_thread_count);
    setThreadIndex(tid, new_thread_index);

    const int32_t choice_value = min + (thread_index / new_thread_count);

    if (choice_value > max) {
        setNumThreads(tid, INVALID_VALUE);
        setThreadIndex(tid, INVALID_VALUE);

        return INVALID_VALUE;
    }

    return choice_value;
}

int64_t _estimate(void (*k)(...), int args) {
    estimation_state = _estimationStateInit(ESTIMATION_THREADS);

    omp_set_num_threads(ESTIMATION_THREADS);
    #pragma omp parallel
    {
        k(args);
    }

    estimation_state->phase = 1;

    int64_t estimated_threads = 1;
    for (int i = 0; i < ESTIMATION_THREADS; i++) {
        int64_t current_estimate = estimation_state->estimated_threads[i];
        estimated_threads = estimated_threads > current_estimate ? estimated_threads : current_estimate;
    }

    return estimated_threads;
}

void explore(void (*k)(...), void *const args, const int num_args) {
    assert(num_args == 1); // all examples have 1 argument
    int *cast_args;
    if (args != NULL) {
        cast_args = (int*) args;
    }

    thread_state = _threadStateInit(MAX_THREADS);
    thread_stats = _threadStatsInit(MAX_THREADS);

    int64_t estimated_threads = _estimate(k, cast_args[0]);
    int64_t kernel_launches = calculateKernelLaunches(MAX_THREADS, estimated_threads);
    int64_t offset = MAX_THREADS;
    int64_t remaining_threads = estimated_threads;

    printf("total threads %ld kernel launches %ld\n", estimated_threads, kernel_launches);

    thread_state->launch_index = 0;
    while (thread_state->launch_index < kernel_launches) {
        int64_t current_threads = remaining_threads < (int64_t) MAX_THREADS ? remaining_threads : (int64_t) MAX_THREADS;
        _threadStateReset(thread_state, current_threads, estimated_threads, (int64_t) thread_state->launch_index * offset);
        
        omp_set_num_threads(current_threads);
        #pragma omp parallel
        {
            k(cast_args[0]);
        }

        thread_state->launch_index++;
    }

    _threadStatsPrint(thread_stats);
}