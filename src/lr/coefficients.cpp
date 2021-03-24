#include <lr/coefficients.hpp>
#include <generic/matrix.hpp>

void set_zero(multi_array<double,2>& a) {
    if(a.sl == stloc::host) {
        fill(a.begin(), a.end(), 0.0);
    } else {
        #ifdef __CUDACC__
        // TODO
        #else
        // ERROR!
        #endif
    }
}


template<size_t d1, size_t d2>
void coeff_double(multi_array<double,d1>& a, multi_array<double,d2>& b, double w, multi_array<double,d1+d2-2>& out) {

  Index N = a.shape()[0];
  int r_a = a.shape()[1];
  int r_b = b.shape()[1];

  double* ptr_j;
  double* ptr_l;
  set_zero(out);
  for(int j = 0; j < r_a; j++){
    for(int l = 0; l < r_b; l++){
      ptr_j = a.extract({j});
      ptr_l = b.extract({l});
      for(int i = 0; i < N; i++){
        out(j,l) += ptr_j[i]*ptr_l[i]*w;
      }
    }
  }
}

template<size_t d1, size_t d2>
void coeff_double_mat(multi_array<double,d1>& a, multi_array<double,d2>& b, double w, multi_array<double,d1+d2-2>& out) {
  matmul_transa(a,b,out);
  out *=  w;
}

template<size_t d1, size_t d2>
void coeff_double_gen(multi_array<double,d1>& a, multi_array<double,d2>& b, double* w, multi_array<double,d1+d2-2>& out) {

  Index N = a.shape()[0];
  int r_a = a.shape()[1];
  int r_b = b.shape()[1];

  double* ptr_j;
  double* ptr_l;
  set_zero(out);
  for(int j = 0; j < r_a; j++){
    for(int l = 0; l < r_b; l++){
      ptr_j = a.extract({j});
      ptr_l = b.extract({l});
      for(int i = 0; i < N; i++){
        out(j,l) += ptr_j[i]*ptr_l[i]*w[i];
      }
    }
  }
}


template<>
void coeff(multi_array<double,2>& a, multi_array<double,2>& b, double w, multi_array<double,2>& out) {
    coeff_double<2,2>(a, b, w, out);
}

template<>
void coeff_mat(multi_array<double,2>& a, multi_array<double,2>& b, double w, multi_array<double,2>& out) {
    coeff_double_mat<2,2>(a, b, w, out);
}

template<>
void coeff_gen(multi_array<double,2>& a, multi_array<double,2>& b, double* w, multi_array<double,2>& out) {
    coeff_double_gen<2,2>(a, b, w, out);
}
