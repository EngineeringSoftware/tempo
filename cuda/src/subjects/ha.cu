
#include "ha.h"
#include <stdio.h>

using namespace std;

__device__ __host__ void HA::resize(int newsize, bool is_insert) {
    // int *new_array = new int[newsize];
    // if (array != nullptr && new_array != nullptr) {
    //     for (int i = 0; i < (is_insert ? size : newsize); i++) {
    //         new_array[i] = array[i];
    //     }
    // }

    // delete [] array;
    // array = new_array;
    arraylength = newsize;
}

__device__ __host__ void HA::insert(int value) {
    if (size == arraylength) {
        resize(size + 1, true);
    }

    array[size] = value;
    int i = size;
    while (true) {
        int parent = (i - 1) / 2;
        if (parent < 0) {
            break;
        }

        if (array[i] > array[parent]) {
            int temp = array[i];
            array[i] = array[parent];
            array[parent] = temp;
            i = parent;
        } else {
            break;
        }
    }
    size++;
}

__device__ __host__ int HA::remove() {
    if (size == 0) {
        return EMPTY_VALUE;
    }

    int ret = array[0];
    array[0] = array[size - 1];
    if (size == arraylength) {
        resize(size - 1, false);
    }

    size--;
    int i = 0;
    while (size != 0) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left >= size) {
            break;
        }

        if (array[i] < array[left]) {
            int temp = array[i];
            array[i] = array[left];
            array[left] = temp;
            i = left;
            continue;
        }

        if (right >= size) {
            break;
        }

        if (array[i] < array[right]) {
            int temp = array[i];
            array[i] = array[right];
            array[right] = temp;
            i = right;
        }
        size--;
    }

    return ret;
}

__device__ __host__ void HA::print() {
    printf("[ size = %d ][ length = %d ] { ", size, arraylength);
    for (int i = 0; i < arraylength; i++) {
        if (i == 0) {
            printf("[ROOT]");
        } else {
            if (i %2 == 1 && (i + 1) < arraylength) {
                printf(array[i] <= array[i + 1] ? "[L, G] " : "[G, L]");
                i++;
            } else {
                printf("[L]");
            }
        }
    }
    printf("}\n");
}
