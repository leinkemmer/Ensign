#pragma once

#include <generic/common.hpp>

// product of an array of numbers
template<class T, size_t d>
T prod(array<T,d> a) {
    T p=1;
    for(size_t i=0;i<d;i++)
        p*=a[i];
    return p;
}

#ifdef __CUDACC__
void* gpu_malloc(size_t size);
#endif
