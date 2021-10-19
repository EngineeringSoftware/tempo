#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "exactthreadscpu.h"

BckState* bck_state = NULL;
BckStats* bck_stats = NULL;

BckState* _bckStateInit(void) {
    BckState* state = (BckState*) malloc(sizeof(BckState));

    state->worklist = (wl_T*) malloc(sizeof(int32_t) * MAX_THREADS * MAX_CHOICES);
    state->buffer = (wl_T*) malloc(sizeof(int32_t) * MAX_WL_SIZE * MAX_CHOICES);

    // initialize first task in buffer
    state->size = 1;
    state->buffer[INDEX(0, 0)] = 0;
    state->buffer[INDEX(0, 1)] = 2;

    return state;
}

void _bckStateDestroy(BckState *state) {
    free(state->worklist);
    free(state->buffer);
    free(state);
}

BckStats* _bckStatsInit(void) {
    BckStats* stats = (BckStats*) malloc(sizeof(BckStats));

    stats->num_tasks_created = 0;
    stats->num_choice_invocations = 0;
    stats->num_kernel_invocations = 0;
    stats->if_counter = 0;
    stats->total_if_counter = 0;

    return stats;
}

void _bckPrintStats(BckStats *stats) {
    #ifdef DEBUG
    printf("# of tasks created     : %u\n", stats->num_tasks_created);
    printf("# of choice invocations: %u\n", stats->num_choice_invocations);
    printf("# of kernel invocations: %u\n", stats->num_kernel_invocations);
    printf("# if_counter: %llu\n", stats->if_counter);
    printf("# total_threads: %llu\n", stats->total_if_counter);
    #endif
}

void cancelThread(int tid) {
    bck_state->worklist[INDEX(tid, 0)] = INVALID_VALUE;
}

int8_t isCancelledThread(int tid) {
    int32_t total = bck_state->worklist[INDEX(tid, 0)];

    return total == INVALID_VALUE;
}

int32_t _choice(int32_t min, int32_t max) {
    const int tid = omp_get_thread_num();
    if (isCancelledThread(tid)) {
        return INVALID_VALUE;
    }

    #ifndef DISABLE_CHOICE_DEBUG
    // increment number of choice invocations
    #pragma omp atomic
    bck_stats->num_choice_invocations++;
    #endif

    if (min == max) {
        return min;
    }

    wl_T* const worklist = bck_state->worklist;
    wl_T* const buffer = bck_state->buffer;

    const wl_T total = worklist[INDEX(tid, 0)];
    const wl_T index = worklist[INDEX(tid, 1)];

    if (index >= (total + 2)) { 
        // Since total is at index 0, and index is at index 1,
        // index will start at 2 by default. This branch is taken
        // when index is pointing to a new _choice call. 

        const int num_choices = max - min + 1;
        int64_t row = 0;
        #pragma omp atomic capture
        {
            row = bck_state->size;
            bck_state->size += num_choices;
        }

        // create new entries
        for (int32_t i = min; i <= max; i++) {
            // atomically get the lock for the next row in the worklist

            #ifndef DISABLE_CHOICE_DEBUG
            if (row >= MAX_WL_SIZE) {
                printf("ERROR: worklist overflow\n");
                exit(EXIT_FAILURE);
            }

            if (INDEX(row, total + 1) > (MAX_WL_SIZE * MAX_CHOICES)) {
                printf("ERROR: Index out of bounds (%d/%d)\n", INDEX(row, total + 1), MAX_WL_SIZE * MAX_CHOICES);
                exit(EXIT_FAILURE);
            }
            #endif

            // copy the current (tid) row to the new row
            buffer[INDEX(row, 0)] = total + 1;
            buffer[INDEX(row, 1)] = 2;

            int j;
            for (j = 2; j < total + 2; j++) {
                // copy the previously selected values of old _choice calls 
                buffer[INDEX(row, j)] = worklist[INDEX(tid, j)];
            }
            buffer[INDEX(row, j)] = i;

            row++;
        }

        cancelThread(tid);

        return INVALID_VALUE;
    } else {
        // printf("tid %d saw before index %d\n", tid);
        // this is an old choice
        // increment index
        worklist[INDEX(tid,1)] = index + 1;
        // printf("returning %d for thread %d\n", worklist[INDEX(tid, index)], tid);
        return worklist[INDEX(tid, index)];
    }
}

void _ignoreIf(int32_t condition) {
    if (condition) {
        int tid = omp_get_thread_num();
        cancelThread(tid);
    }
}

void _countIf(int32_t condition) {
    #ifndef DISABLE_CHOICE_DEBUG
    int tid = omp_get_thread_num();
    if (isCancelledThread(tid)) {
        return;
    }

    #pragma omp atomic
    bck_stats->total_if_counter++;

    if (condition) {
        #pragma omp atomic
        bck_stats->if_counter++;
    }
    #endif
}

void explore(void (*k)(...), void *const args, const int num_args) {
    assert(num_args == 1); // all examples have 1 argument
    int *cast_args;
    if (args != NULL) {
        cast_args = (int*) args;
    }

    bck_state = _bckStateInit();
    bck_stats = _bckStatsInit();

    int total_size = 0;
    while (bck_state->size > 0) {
        int64_t current_threads = min(bck_state->size, (int64_t) MAX_THREADS);
        int buffer_start = (bck_state->size - current_threads) * MAX_CHOICES;

        memcpy(bck_state->worklist, &bck_state->buffer[buffer_start], current_threads * MAX_CHOICES * sizeof(int32_t));

        bck_state->size -= current_threads;

        bck_stats->num_kernel_invocations++;
        bck_stats->num_tasks_created += current_threads;

        omp_set_num_threads(current_threads);

        #pragma omp parallel
        {
            k(cast_args[0]);
        }

    }

    _bckPrintStats(bck_stats);

    _bckStateDestroy(bck_state);
    free(bck_stats);
}
