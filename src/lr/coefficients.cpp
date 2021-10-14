#include <lr/coefficients.hpp>

template<class T>
void coeff(const multi_array<T,2>& a, const multi_array<T,2>& b, T w, multi_array<T,2>& out, const blas_ops& blas) {
  blas.matmul_transa(a,b,out);
  if(a.sl == stloc::host){
    out *=  w;
  }else{
    #ifdef __CUDACC__
      ptw_mult_scal<<<(out.num_elements()+n_threads-1)/n_threads,n_threads>>>(out.num_elements(), out.begin(), w);
    #else
      cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
    #endif
  }

}
template void coeff(const multi_array<double,2>& a, const multi_array<double,2>& b, double w, multi_array<double,2>& out, const blas_ops& blas);
template void coeff(const multi_array<float,2>& a, const multi_array<float,2>& b, float w, multi_array<float,2>& out, const blas_ops& blas);


template<class T>
void coeff(const multi_array<T,2>& a, const multi_array<T,2>& b, const multi_array<T,1>& w, multi_array<T,2>& out, const blas_ops& blas) {
  multi_array<T,2> tmp(b.shape(),b.sl);
  if(b.sl == stloc::host){
    ptw_mult_row(b,w,tmp);
  }else{
    #ifdef __CUDACC__
      ptw_mult_row_k<<<(b.num_elements()+n_threads-1)/n_threads,n_threads>>>(b.num_elements(), b.shape()[0], b.begin(), w.begin(), tmp.begin());
    #else
      cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
    #endif
  }

  blas.matmul_transa(a,tmp,out);
}
template void coeff(const multi_array<double,2>& a, const multi_array<double,2>& b, const multi_array<double,1>& w, multi_array<double,2>& out, const blas_ops& blas);
template void coeff(const multi_array<float,2>& a, const multi_array<float,2>& b, const multi_array<float,1>&, multi_array<float,2>& out, const blas_ops& blas);


template<class T>
void integrate(const multi_array<T,2>& a, const multi_array<T,1>& w, multi_array<T,1>& out, const blas_ops& blas) {
  blas.matvec_trans(a,w,out);
}
template void integrate(const multi_array<double,2>& a, const multi_array<double,1>& w, multi_array<double,1>& out, const blas_ops& blas);
template void integrate(const multi_array<float,2>& a, const multi_array<float,1>& w, multi_array<float,1>& out, const blas_ops& blas);


template<class T>
void integrate(const multi_array<T,2>& a, T w, multi_array<T,1>& out, const blas_ops& blas) {
  multi_array<T,1> vec({a.shape()[0]}, a.sl);
  set_const(vec,w);
  blas.matvec_trans(a,vec,out);
}
template void integrate(const multi_array<double,2>& a, double w, multi_array<double,1>& out, const blas_ops& blas);
template void integrate(const multi_array<float,2>& a, float w, multi_array<float,1>& out, const blas_ops& blas);
