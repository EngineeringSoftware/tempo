
// The purpose of this file is to include the correct
// library without the user having to use the ifdef
// macros in their code

#ifndef EXPLORE_H
#define EXPLORE_H

#define DEVICE __device__

#ifdef EXACTTHREADS
#include "exactthreads.h"
#elif defined(CONSTTHREADS)
#include "constthreads.h"
#elif defined(ALLTHREADS)
#include "allthreads.h"

#else
#undef DEVICE
#define DEVICE __host__
#ifdef ALLTHREADSOMP
#include "allthreadscpu.h"
#elif defined(EXACTTHREADSOMP)
#include "exactthreadscpu.h"
#endif

#endif


/*
 * Non-deterministically choose values between min and max and
 * schedule new threads.
 *
 * @public
 */
DEVICE int _choice(int min, int max);

/*
 * Stop execution of the current thread if the given argument is true.
 *
 * @public
 */
DEVICE void _ignoreIf(int32_t);

/*
 * Increment underling counter if the given argument is true.
 *
 * @public
 */
DEVICE void _countIf(int32_t);

/*
 * Explore a kernel, passing in the arguments and number of arguments
 *
 * @public
 */
void explore(void (*k)(...), void*, int);


#endif
