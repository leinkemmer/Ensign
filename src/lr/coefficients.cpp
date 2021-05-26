#include <lr/coefficients.hpp>

template<class T>
void coeff(multi_array<T,2>& a, multi_array<T,2>& b, T w, multi_array<T,2>& out) {
  matmul_transa(a,b,out);
  if(a.sl == stloc::host){
    out *=  w;
  }else{
    #ifdef __CUDACC__
      ptw_mult_scal<<<2,2>>>(out.num_elements(), out.begin(), w);
    #else
      cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
    #endif
  }

}
template void coeff(multi_array<double,2>& a, multi_array<double,2>& b, double w, multi_array<double,2>& out);
template void coeff(multi_array<float,2>& a, multi_array<float,2>& b, float w, multi_array<float,2>& out);
/*
template<class T, size_t d1, size_t d2>
void coeff_T(multi_array<T,d1>& a, multi_array<T,d2>& b, T w, multi_array<T,d1+d2-2>& out) {
  matmul_transa(a,b,out);
  out *=  w;
}
*/
template<class T>
void coeff(multi_array<T,2>& a, multi_array<T,2>& b, T* w, multi_array<T,2>& out) {
  multi_array<T,2> tmp(b.shape(),b.sl);
  if(b.sl == stloc::host){
    ptw_mult_row(b,w,tmp);
  }else{
    #ifdef __CUDACC__
      ptw_mult_row_k<<<64,64>>>(b.num_elements(), b.shape()[0], b.begin(), w, tmp.begin());
    #else
      cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
    #endif
  }

  matmul_transa(a,tmp,out);
}
template void coeff(multi_array<double,2>& a, multi_array<double,2>& b, double* w, multi_array<double,2>& out);
template void coeff(multi_array<float,2>& a, multi_array<float,2>& b, float* w, multi_array<float,2>& out);
/*
template<class T, size_t d1, size_t d2>
void coeff_T(multi_array<T,d1>& a, multi_array<T,d2>& b, T* w, multi_array<T,d1+d2-2>& out) {
  multi_array<T,d2> tmp(b.shape());
  ptw_mult_row(b,w,tmp);
  matmul_transa(a,tmp,out);
}
*/
template<class T>
void coeff_one(multi_array<T,2>& a, multi_array<T,1>& w, multi_array<T,1>& out) {
  matvec_trans(a,w,out);
}
template void coeff_one(multi_array<double,2>& a, multi_array<double,1>& w, multi_array<double,1>& out);
template void coeff_one(multi_array<float,2>& a, multi_array<float,1>& w, multi_array<float,1>& out);
/*
template<class T, size_t d1>
void coeff_one_T(multi_array<T,d1>& a, T* w, multi_array<T,1>& out) {
  for(Index i = 0; i < a.shape()[1]; i++){
    out(i) = T(0.0);
    for(Index j = 0; j < a.shape()[0]; j++){
      out(i) += a(j,i)*w[j];
    }
  }
}
*/

template<class T>
void coeff_one(multi_array<T,2>& a, T w, multi_array<T,1>& out) {
  multi_array<T,1> vec({a.shape()[0]}, a.sl);
  set_const(vec,w);
  matvec_trans(a,vec,out);
}
template void coeff_one(multi_array<double,2>& a, double w, multi_array<double,1>& out);
template void coeff_one(multi_array<float,2>& a, float w, multi_array<float,1>& out);

/*
template<class T, size_t d1>
void coeff_one_T(multi_array<T,d1>& a, T w, multi_array<T,1>& out) {
  for(Index i = 0; i < a.shape()[1]; i++){
    out(i) = T(0.0);
    for(Index j = 0; j < a.shape()[0]; j++){
      out(i) += a(j,i);
    }
  }
  out *= w;
}
*/
/*
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

template<>
void coeff_one(multi_array<double,2>& a, double* w, multi_array<double,1>& out) {
    coeff_one_T<double,2>(a, w, out);
}
template<>
void coeff_one(multi_array<float,2>& a, float* w, multi_array<float,1>& out) {
    coeff_one_T<float,2>(a, w, out);
}

template<>
void coeff_one(multi_array<double,2>& a, double w, multi_array<double,1>& out) {
    coeff_one_T<double,2>(a, w, out);
}
template<>
void coeff_one(multi_array<float,2>& a, float w, multi_array<float,1>& out) {
    coeff_one_T<float,2>(a, w, out);
}
*/
