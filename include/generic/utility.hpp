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

// phi1 of scalar (complex float or double)
/*
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
  if(a == complex<double>(0.0,0.0)){
    p = 1.0;
  }else{
    p = (exp(a) - 1.0) / a;
  }
  }
  return p;
}
*/

template<class T>
T phi1(T a) {
  return (abs(a) < 1e-7) ? complex<double>(1.0 + a.real(), a.imag()) : exp(a/2.0)*sinh(a/2.0)/(a/2.0);
}

template<class T>
T phi1_im(T a) { // use it only for purely imaginary
  if(abs(a.imag()) < 1e-7){
    return complex<double>(1.0+a.imag(),a.imag());
  }else{
    return complex<double>(sin(a.imag())/a.imag(),2*pow(sin(a.imag()/2.0),2)/a.imag());
  }
}

template<class T>
T phi2_im(T a) { // use it only for purely imaginary
  if(abs(a.imag()) < 1e-7){
    return complex<double>(0.5+a.imag(),a.imag());
  }else{
    return complex<double>(2*pow(sin(a.imag()/2.0),2)/pow(a.imag(),2),(a.imag()-sin(a.imag()))/pow(a.imag(),2));
  }
}

#ifdef __CUDACC__
void* gpu_malloc(size_t size);
#endif
