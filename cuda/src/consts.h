
#ifndef CONSTS_H
#define CONSTS_H

#define FALSE 0
#define TRUE 1

// Size of the pool for objects (UDITA)
#define POOL_SIZE 12

#define RBT_POOL_SIZE 14

#define NQUEENS_POOL_SIZE 16

// Max size of input worklist
#ifndef MAX_IN_WL
#define MAX_IN_WL 8192 * 5
#endif
// Max size of output worklist
#define MAX_OUT_WL 8192 * 5 * 100

#ifndef ALL_THREADS
#define ALL_THREADS 500000000
#endif

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 1024
#endif

// Number of blocks that has to be used when invoking EXPLORE
extern int32_t starting_blocks;

// Number of threads that has to be used when invoking EXPLORE
extern int32_t starting_threads;

#endif