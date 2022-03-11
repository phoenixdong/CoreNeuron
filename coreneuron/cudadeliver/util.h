#pragma once

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static inline int alignUp(const int a, const int b) {
    assert(a >= 0 && b > 0);
    return ((a + b - 1) / b) * b;
}

static inline int roundedAdd(const int a, const int b, const int c) {
    return (a + c) % b;
}

static inline int roundedInc(const int a, const int b) {
    return roundedAdd(a, b, 1);
}

static inline double sizeInKb(const size_t size) {
    return (double)size / 1024;
}

static inline double sizeInMb(const size_t size) {
    return (double)size / 1024 / 1024;
}

static inline double sizeInGb(const size_t size) {
    return (double)size / 1024 / 1024 / 1024;
}


// #define CUDA_DELIVER_DEBUG
