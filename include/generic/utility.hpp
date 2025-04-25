#pragma once

#include <generic/common.hpp>

namespace Ensign {

// product of an array of numbers
template<class T, size_t d>
T prod(array<T,d> a) {
  T p=1;
  for(size_t i=0;i<d;i++)
  p*=a[i];
  return p;
}

/* Computes (exp(z)-1)/z
*
* There is a removeable singularity at z=0.
*/
template<class T>
T phi1(T a) {
  return (abs(a) < 1e-7) ? complex<double>(1.0 + a.real(), a.imag()) : exp(a/2.0)*sinh(a/2.0)/(a/2.0);
}

/* Computes (exp(i a)-1)/(i a)
*
* There is a removeable singularity at a=0.
*/
template<class T>
T phi1_im(T a) { // use it only for purely imaginary
  if(abs(a.imag()) < 1e-7){
    return complex<double>(1.0+a.imag(),a.imag());
  }else{
    return complex<double>(sin(a.imag())/a.imag(),2*pow(sin(a.imag()/2.0),2)/a.imag());
  }
}

/* Computes (exp(i a)-1-i a)/((i a)^2)
*
* There is a removeable singularity at a=0.
*/
template<class T>
T phi2_im(T a) { // use it only for purely imaginary
  if(abs(a.imag()) < 1e-7){
    return complex<double>(0.5+a.imag(),a.imag());
  }else{
    return complex<double>(2*pow(sin(a.imag()/2.0),2)/pow(a.imag(),2),(a.imag()-sin(a.imag()))/pow(a.imag(),2));
  }
}

#ifdef __CUDACC__
/* Typesafe wrapper for cudaMalloc
*/
void* gpu_malloc(size_t size);
#endif


/* Parser for command line arguments separated by whitespaces
*/
template<size_t d>
array<Index,d> parse(string s) {
    array<Index,d> out;
    std::istringstream iss(s);
    for(size_t i=0;i<d;i++) {
        if(iss.eof()) {
            cout << "ERROR: not enough dof provided to parse" << endl;
            exit(1);
        }
        iss >> out[i];
    }

    if(!iss.eof()) {
        cout << "ERROR: to many dof provided to parse" << endl;
        exit(1);
    }
    return out;
}

/* Maximum function which treats NaNs correctly
*/
template<class T>
double max_err(T err, T diff) {
  if(std::isnan(diff)) 
    return std::numeric_limits<T>::infinity();
  else
    return max(err, diff);
}

template <class InputIt, class OutputIt>
void remove_element(InputIt first, InputIt last, OutputIt d_first, const Index idx)
{
    std::copy(first, first + idx, d_first);
    std::copy(first + idx + 1, last, d_first + idx);
}

} // namespace Ensign