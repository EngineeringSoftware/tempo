
/*
 * This is C++ version of
 * predicate.heaparray.HeapArray_pred
 * from http://mir.cs.illinois.edu/coverage.
 */

#ifndef HA_H_
#define HA_H_

class HA {
    const int EMPTY_VALUE = -1;

    /* element on the heap */
    int *array;

    /* capacity */
    int arraylength;

    /* number of actual elements */
    int size;

    __device__ __host__ void resize(int newsize, bool is_insert);

 public:

    __device__ __host__ HA(int *array) : size(0), array(array), arraylength(0) {}
    /* __device__ __host__ ~HA() { delete [] array; } */

    __device__ __host__ int getSize() { return size; }
    __device__ __host__ void insert(int value);
    __device__ __host__ int remove();
    __device__ __host__ void print();
};

#endif
