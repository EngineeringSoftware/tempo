
#ifndef ERRORS_H
#define ERRORS_H

void checkCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) checkCudaErrorAux(__FILE__,__LINE__, #value, value)

#define CUDA_MALLOC(...) CUDA_CHECK_RETURN(cudaMalloc(__VA_ARGS__))
#define CUDA_MALLOC_MANAGED(...) CUDA_CHECK_RETURN(cudaMallocManaged(__VA_ARGS__))
#define CUDA_FREE(...) CUDA_CHECK_RETURN(cudaFree(__VA_ARGS__))
#define CUDA_MEMCPY_TO_SYMBOL(...) CUDA_CHECK_RETURN(cudaMemcpyToSymbol(__VA_ARGS__))

#endif
