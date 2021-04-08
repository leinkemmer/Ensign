#include <generic/matrix.hpp>
#include <cblas.h>
#ifdef __CUDACC__
#endif

template<class T>
void set_zero(multi_array<T,2>& a) {
  if(a.sl == stloc::host) {
    fill(a.begin(), a.end(), T(0.0));
  } else {
    #ifdef __CUDACC__
    multi_array<T,2> _a(a.shape());
    fill(_a.begin(), _a.end(), T(0.0));
    a = _a;
    #else
    cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
    << __LINE__ << endl;
    exit(1);
    #endif
  }
}
template void set_zero(multi_array<double,2>&);
template void set_zero(multi_array<float,2>&);
template void set_zero(multi_array<complex<double>,2>&);

template<class T>
void set_identity(multi_array<T,2>& a){
  if(a.sl == stloc::host){
    set_zero(a);
    for(Index i = 0; i<a.shape()[0]; i++){
      a(i,i) = T(1.0);
    }
  } else {
    #ifdef __CUDACC__
    multi_array<T,2> _a(a.shape());
    set_zero(_a);
    for(Index i = 0; i<_a.shape()[0]; i++){
      _a(i,i) = T(1.0);
    }
    a = _a;
    #else
    cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
    << __LINE__ << endl;
    exit(1);
    #endif
  }
}
template void set_identity(multi_array<double,2>&);
template void set_identity(multi_array<float,2>&);

template<class T> // we should write a kernel for this one, will be performed on the GPU
void ptw_mult_col(multi_array<T,2>& a, T* w, multi_array<T,2>& out){
  Index N = a.shape()[0];
  for(int r = 0; r < a.shape()[1]; r++){
    T* ptr = a.extract({r});
    for(Index i = 0; i < a.shape()[0]; i++){
      out(i,r) = ptr[i]*w[i];
    }
  }
}
template void ptw_mult_col(multi_array<double,2>&, double*, multi_array<double,2>&);
template void ptw_mult_col(multi_array<float,2>&, float*, multi_array<float,2>&);

template<>
void matmul(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      a.shape()[0], b.shape()[1], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), a.shape()[1], 0.0,
      c.begin(), a.shape()[0]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      cublasHandle_t  handle;
      cublasCreate (&handle);
      double alpha = 1.0;
      double beta = 0.0;

      cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
        a.shape()[0],b.shape()[1],a.shape()[1],
        &alpha,a.begin(),a.shape()[0],
        b.begin(),a.shape()[1],&beta,
        c.begin(), a.shape()[0]);
        #else
        cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
        #endif
      } else {
        cout << "ERROR: inputs and output must be all on CPU or on GPU" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
      }
}

template<>
void matmul(const multi_array<complex<double>,2>& a, const multi_array<complex<double>,2>& b, multi_array<complex<double>,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    complex<double> one(1.0,0.0);
    complex<double> zero(0.0,0.0);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      a.shape()[0], b.shape()[1], a.shape()[1],
      &one, a.begin(), a.shape()[0],
      b.begin(), a.shape()[1], &zero,
      c.begin(), a.shape()[0]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      // TO DO
      } else {
        // TO DO
      }

}

template<>
void matmul(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      a.shape()[0], b.shape()[1], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), a.shape()[1], 0.0,
      c.begin(), a.shape()[0]);
  }else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
    #ifdef __CUDACC__
    cublasHandle_t  handle;
    cublasCreate (&handle);
    float alpha = 1.0;
    float beta = 0.0;

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
        a.shape()[0],b.shape()[1],a.shape()[1],
        &alpha,a.begin(),a.shape()[0],
        b.begin(),a.shape()[1],&beta,
        c.begin(), a.shape()[0]);
    #else
      cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
    #endif
  } else {
      cout << "ERROR: inputs and output must be all on CPU or on GPU" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
  }

}

template<>
void matmul_transa(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      a.shape()[1], b.shape()[1], a.shape()[0],
      1.0, a.begin(), a.shape()[0],
      b.begin(), a.shape()[0], 0.0,
      c.begin(), a.shape()[1]);
  } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
    #ifdef __CUDACC__
    cublasHandle_t  handle;
    cublasCreate (&handle);
    double alpha = 1.0;
    double beta = 0.0;

    cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,
      a.shape()[1],b.shape()[1],a.shape()[0],
      &alpha,a.begin(),a.shape()[0],
      b.begin(),a.shape()[0],&beta,
      c.begin(), a.shape()[1]);
    #else
      cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
    #endif
  } else {
    cout << "ERROR: inputs and output must be all on CPU or on GPU" << __FILE__ << ":"
    << __LINE__ << endl;
    exit(1);
  }

}
template<>
void matmul_transa(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      a.shape()[1], b.shape()[1], a.shape()[0],
      1.0, a.begin(), a.shape()[0],
      b.begin(), a.shape()[0], 0.0,
      c.begin(), a.shape()[1]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      cublasHandle_t  handle;
      cublasCreate (&handle);
      float alpha = 1.0;
      float beta = 0.0;

      cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,
        a.shape()[1],b.shape()[1],a.shape()[0],
        &alpha,a.begin(),a.shape()[0],
        b.begin(),a.shape()[0],&beta,
        c.begin(), a.shape()[1]);
        #else
        cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
        #endif
      } else {
        cout << "ERROR: inputs and output must be all on CPU or on GPU" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
      }

}
template<>
void matmul_transb(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      a.shape()[0], b.shape()[0], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], 0.0,
      c.begin(), a.shape()[0]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      cublasHandle_t  handle;
      cublasCreate (&handle);
      double alpha = 1.0;
      double beta = 0.0;

      cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
        a.shape()[0],b.shape()[0],a.shape()[1],
        &alpha,a.begin(),a.shape()[0],
        b.begin(),b.shape()[0],&beta,
        c.begin(), a.shape()[0]);
        #else
        cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
        #endif
      } else {
        cout << "ERROR: inputs and output must be all on CPU or on GPU" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
      }
}
template<>
void matmul_transb(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      a.shape()[0], b.shape()[0], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], 0.0,
      c.begin(), a.shape()[0]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      cublasHandle_t  handle;
      cublasCreate (&handle);
      float alpha = 1.0;
      float beta = 0.0;

      cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
        a.shape()[0],b.shape()[0],a.shape()[1],
        &alpha,a.begin(),a.shape()[0],
        b.begin(),b.shape()[0],&beta,
        c.begin(), a.shape()[0]);
        #else
        cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
        #endif
      } else {
        cout << "ERROR: inputs and output must be all on CPU or on GPU" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
      }
}
template<>
void matmul_transb(const multi_array<complex<double>,2>& a, const multi_array<complex<double>,2>& b, multi_array<complex<double>,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    complex<double> one(1.0,0.0);
    complex<double> zero(0.0,0.0);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      a.shape()[0], b.shape()[0], a.shape()[1],
      &one, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], &zero,
      c.begin(), a.shape()[0]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      // TO DO
      } else {
        // TO DO
      }
}

template<>
void matmul_transab(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
      a.shape()[1], b.shape()[0], a.shape()[0],
      1.0, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], 0.0,
      c.begin(), a.shape()[1]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      cublasHandle_t  handle;
      cublasCreate (&handle);
      double alpha = 1.0;
      double beta = 0.0;

      cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,
        a.shape()[1],b.shape()[0],a.shape()[0],
        &alpha,a.begin(),a.shape()[0],
        b.begin(),b.shape()[0],&beta,
        c.begin(), a.shape()[1]);
        #else
        cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
        #endif
      } else {
        cout << "ERROR: inputs and output must be all on CPU or on GPU" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
      }
}

template<>
void matmul_transab(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c){
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans,
      a.shape()[1], b.shape()[0], a.shape()[0],
      1.0, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], 0.0,
      c.begin(), a.shape()[1]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      cublasHandle_t  handle;
      cublasCreate (&handle);
      float alpha = 1.0;
      float beta = 0.0;

      cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,
        a.shape()[1],b.shape()[0],a.shape()[0],
        &alpha,a.begin(),a.shape()[0],
        b.begin(),b.shape()[0],&beta,
        c.begin(), a.shape()[1]);
        #else
        cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
        #endif
      } else {
        cout << "ERROR: inputs and output must be all on CPU or on GPU" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
      }

}

//template<class T>
//void matmul_transa(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

//template<class T>
//void matmul_transb(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);


//template struct multi_array<double,2>;
//template struct multi_array<float,2>;

//template void set_zero(multi_array<double,2>&);
//template void set_zero(multi_array<float,2>&);
