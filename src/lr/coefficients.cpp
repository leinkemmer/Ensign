#include <generic/matrix.hpp>
#include <lr/coefficients.hpp>

template<class T, size_t d1, size_t d2>
void coeff_T(multi_array<T,d1>& a, multi_array<T,d2>& b, T w, multi_array<T,d1+d2-2>& out) {
  matmul_transa(a,b,out);
  out *=  w;
}

template<class T, size_t d1, size_t d2>
void coeff_T(multi_array<T,d1>& a, multi_array<T,d2>& b, T* w, multi_array<T,d1+d2-2>& out) {
  multi_array<T,d2> tmp(b.shape());
  ptw_mult_col(b,w,tmp);
  matmul_transa(a,tmp,out);
}

template<>
void coeff(multi_array<double,2>& a, multi_array<double,2>& b, double w, multi_array<double,2>& out) {
    coeff_T<double,2,2>(a, b, w, out);
}
template<>
void coeff(multi_array<float,2>& a, multi_array<float,2>& b, float w, multi_array<float,2>& out) {
    coeff_T<float,2,2>(a, b, w, out);
}

template<>
void coeff(multi_array<double,2>& a, multi_array<double,2>& b, double* w, multi_array<double,2>& out) {
    coeff_T<double,2,2>(a, b, w, out);
}
template<>
void coeff(multi_array<float,2>& a, multi_array<float,2>& b, float* w, multi_array<float,2>& out) {
    coeff_T<float,2,2>(a, b, w, out);
}
