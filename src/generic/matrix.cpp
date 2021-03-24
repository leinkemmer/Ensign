#include <generic/matrix.hpp>
#include <cblas.h>

template<>
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

template<>
void set_zero(multi_array<float,2>& a) {
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

template<class T>
void set_identity(multi_array<T,2>& a){
  set_zero(a);
  for(int i = 0; i<a.shape()[0]; i++){
    a(i,i) = T(1.0);
  }
}
template void set_identity(multi_array<double,2>&);
template void set_identity(multi_array<float,2>&);
/*
template<>
void set_identity(multi_array<float,2>& a){
  set_zero(a);
  for(int i = 0; i<a.shape()[0]; i++){
    a(i,i) = 1.0;
  }
}*/

template<>
void matmul(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              a.shape()[0], b.shape()[1], a.shape()[1],
              1.0, a.begin(), a.shape()[0],
              b.begin(), a.shape()[1], 0.0,
              c.begin(), a.shape()[0]);

}

template<>
void matmul_transa(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              a.shape()[1], b.shape()[1], a.shape()[0],
              1.0, a.begin(), a.shape()[0],
              b.begin(), a.shape()[0], 0.0,
              c.begin(), a.shape()[1]);

}

template<>
void matmul_transb(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
              a.shape()[0], b.shape()[0], a.shape()[1],
              1.0, a.begin(), a.shape()[0],
              b.begin(), b.shape()[0], 0.0,
              c.begin(), a.shape()[0]);

}

template<>
void matmul_transab(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){

  cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
              a.shape()[1], b.shape()[0], a.shape()[0],
              1.0, a.begin(), a.shape()[0],
              b.begin(), b.shape()[0], 0.0,
              c.begin(), a.shape()[1]);

}
/*
#ifdef __CUDACC__
template<>
void matmul_gpu(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){
  cublasHandle_t  handle;
  cublasCreate (&handle);
  double alpha = 1.0;
  double beta = 0.0;

  cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
              a.shape()[0],b.shape()[1],a.shape()[1],
              &alpha,a.begin(),a.shape()[0],
              b.begin(),a.shape()[1],&beta,
              c.begin(), a.shape()[0]);

}
#endif
*/

//template<class T>
//void matmul_transa(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

//template<class T>
//void matmul_transb(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);


//template struct multi_array<double,2>;
//template struct multi_array<float,2>;

//template void set_zero(multi_array<double,2>&);
//template void set_zero(multi_array<float,2>&);
