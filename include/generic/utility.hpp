#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>

// product of an array of numbers
template<class T, size_t d>
T prod(array<T,d> a) {
  T p=1;
  for(size_t i=0;i<d;i++)
  p*=a[i];
  return p;
}

// phi1 of scalar (complex float or double)
template<class T>
T phi1(T a) {
  T p = T(1.0);
  T fact = 1.0;
  if (abs(a)<1){
    for(int i = 1; i < 18; i++){
      fact *= (i+1);
      p += (1.0/fact)*pow(a,i);
    }
  } else{
    p = (exp(a) - 1.0) / a;
  }
  return p;
}


#ifdef __CUDACC__
void* gpu_malloc(size_t size);
#endif
