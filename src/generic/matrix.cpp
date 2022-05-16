#include <generic/matrix.hpp>
#include <generic/timer.hpp>


#ifndef __MKL__
extern "C" {
  extern int dgees_(char*,char*,void*,int*,double*,int*, int*, double*, double*, double*, int*, double*, int*, bool*,int*);
}
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
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index i = 0; i<a.shape()[0]; i++){
      a(i,i) = T(1.0);
    }
  } else {
    #ifdef __CUDACC__
    multi_array<T,2> _a(a.shape());
    set_zero(_a);
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
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

template<class T, size_t d>
void set_const(multi_array<T,d>& a,T alpha) {
  if(a.sl == stloc::host) {
    fill(a.begin(), a.end(), alpha);
  } else {
    #ifdef __CUDACC__
      fill_gpu<<<(a.num_elements()+n_threads-1)/n_threads,n_threads>>>(a.num_elements(),a.begin(),alpha);
    #else
    cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
    << __LINE__ << endl;
    exit(1);
    #endif
  }
}
template void set_const(multi_array<double,1>&, double);
template void set_const(multi_array<float,1>&, float);
template void set_const(multi_array<double,2>&, double);
template void set_const(multi_array<float,2>&, float);


template<class T>
void ptw_mult_row(const multi_array<T,2>& a, const multi_array<T,1>& w, multi_array<T,2>& out){
  if(a.sl == stloc::host){
    #ifdef __OPENMP__
    #pragma omp parallel for collapse(2)
    for(int r = 0; r < a.shape()[1]; r++){
      for(Index i = 0; i < a.shape()[0]; i++){
        const T* ptr = a.extract({r});
        out(i,r) = ptr[i]*w(i);
      }
    }
    #else
    for(int r = 0; r < a.shape()[1]; r++){
      const T* ptr = a.extract({r});
      for(Index i = 0; i < a.shape()[0]; i++){
        out(i,r) = ptr[i]*w(i);
      }
    }
    #endif
  }else{
    #ifdef __CUDACC__
    if(std::is_same<T, double>::value) {
      ptw_mult_row_k<<<(a.num_elements()+n_threads-1)/n_threads,n_threads>>>(a.num_elements(),a.shape()[0],a.begin(),w.begin(),out.begin());
    } else if(std::is_same<T, complex<double>>::value) {
      ptw_mult_row_k<<<(a.num_elements()+n_threads-1)/n_threads,n_threads>>>(a.num_elements(),a.shape()[0],(cuDoubleComplex*)a.begin(),(cuDoubleComplex*)w.begin(),(cuDoubleComplex*)out.begin());
    } else {
      cout << "ERROR: ptw_mult_row for not implemented for that datatype" << endl;
      exit(1);
    }
    #else
    cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
    << __LINE__ << endl;
    exit(1);
    #endif
  }

}
template void ptw_mult_row(const multi_array<double,2>&, const multi_array<double,1>&, multi_array<double,2>&);
template void ptw_mult_row(const multi_array<float,2>&, const multi_array<float,1>&, multi_array<float,2>&);
template void ptw_mult_row(const multi_array<complex<double>,2>&, const multi_array<complex<double>,1>&, multi_array<complex<double>,2>&);
template void ptw_mult_row(const multi_array<complex<float>,2>&, const multi_array<complex<float>,1>&, multi_array<complex<float>,2>&);


template<class T>
void transpose_inplace(multi_array<T,2>& a){
  if (a.sl == stloc::host){
  Index m = a.shape()[0];
  T tmp = 0.0;
  for(Index r = 0; r < m; r++){
    for(Index i = 1 + r; i < m; i++){
      tmp = a(r,i);
      a(r,i) = a(i,r);
      a(i,r) = tmp;
    }
  }
  } else if (a.sl == stloc::device){
    #ifdef __CUDACC__
    transpose_inplace<<<a.num_elements(),1>>>(a.shape()[0],a.begin()); 
    #else
      cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
    #endif
  } else {
    cout << "ERROR: input must be on CPU or on GPU" << __FILE__ << ":"
    << __LINE__ << endl;
    exit(1);
  }
}

template void transpose_inplace(multi_array<double,2>&);
//template void transpose_inplace(multi_array<float,2>&);


blas_ops::blas_ops(bool _gpu) : gpu(_gpu), handle(0), handle_devres(0) {
  #ifdef __CUDACC__
  if(gpu) {
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cout << "ERROR: cublasCreate failed. Error code: " << status << endl;
      exit(1);
    }

    cublasCreate (&handle_devres);
    cublasSetPointerMode(handle_devres, CUBLAS_POINTER_MODE_DEVICE);
  }
  #endif
}

blas_ops::~blas_ops() {
  #ifdef __CUDACC__
  if(handle)
      cublasDestroy(handle);
 
  if(handle_devres)
      cublasDestroy(handle_devres);
  #endif
}

template<>
void blas_ops::matmul(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      a.shape()[0], b.shape()[1], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), a.shape()[1], 0.0,
      c.begin(), a.shape()[0]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
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
void blas_ops::matmul(const multi_array<complex<double>,2>& a, const multi_array<complex<double>,2>& b, multi_array<complex<double>,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    complex<double> one(1.0,0.0);
    complex<double> zero(0.0,0.0);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      a.shape()[0], b.shape()[1], a.shape()[1],
      &one, a.begin(), a.shape()[0],
      b.begin(), a.shape()[1], &zero,
      c.begin(), a.shape()[0]);
  }
  else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
    #ifdef __CUDACC__
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
            a.shape()[0],b.shape()[1],a.shape()[1],
            &one,(cuDoubleComplex*)a.begin(),a.shape()[0],
            (cuDoubleComplex*)b.begin(), a.shape()[1],&zero,
            (cuDoubleComplex*)c.begin(), a.shape()[0]);
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
void blas_ops::matmul(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      a.shape()[0], b.shape()[1], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), a.shape()[1], 0.0,
      c.begin(), a.shape()[0]);
  }else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
    #ifdef __CUDACC__
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


#ifdef __CUDACC__
template<>
void blas_ops::matmul(const multi_array<cuDoubleComplex,2>& a, const multi_array<cuDoubleComplex,2>& b, multi_array<cuDoubleComplex,2>& c) const {

  cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

  cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
          a.shape()[0],b.shape()[1],a.shape()[1],
          &one,a.begin(),a.shape()[0],
          b.begin(),a.shape()[1],&zero,
          c.begin(), a.shape()[0]);

}
#endif

template<>
void blas_ops::matmul_transa(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      a.shape()[1], b.shape()[1], a.shape()[0],
      1.0, a.begin(), a.shape()[0],
      b.begin(), a.shape()[0], 0.0,
      c.begin(), a.shape()[1]);
  } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
    #ifdef __CUDACC__
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
void blas_ops::matmul_transa(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      a.shape()[1], b.shape()[1], a.shape()[0],
      1.0, a.begin(), a.shape()[0],
      b.begin(), a.shape()[0], 0.0,
      c.begin(), a.shape()[1]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
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
void blas_ops::matmul_transb(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      a.shape()[0], b.shape()[0], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], 0.0,
      c.begin(), a.shape()[0]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
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
void blas_ops::matmul_transb(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      a.shape()[0], b.shape()[0], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], 0.0,
      c.begin(), a.shape()[0]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
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
void blas_ops::matmul_transb(const multi_array<complex<double>,2>& a, const multi_array<complex<double>,2>& b, multi_array<complex<double>,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    complex<double> one(1.0,0.0);
    complex<double> zero(0.0,0.0);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      a.shape()[0], b.shape()[0], a.shape()[1],
      &one, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], &zero,
      c.begin(), a.shape()[0]);
  } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
      cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

      cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
        a.shape()[0],b.shape()[0],a.shape()[1],
        &one, (cuDoubleComplex*)a.begin(), a.shape()[0],
        (cuDoubleComplex*)b.begin(), b.shape()[0],&zero,
        (cuDoubleComplex*)c.begin(), a.shape()[0]);
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

#ifdef __CUDACC__
template<>
void blas_ops::matmul_transb(const multi_array<cuDoubleComplex,2>& a, const multi_array<cuDoubleComplex,2>& b, multi_array<cuDoubleComplex,2>& c) const {
        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

        cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
          a.shape()[0],b.shape()[0],a.shape()[1],
          &one,a.begin(),a.shape()[0],
          b.begin(),b.shape()[0],&zero,
          c.begin(), a.shape()[0]);

}
#endif

template<>
void blas_ops::matmul_transab(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
      a.shape()[1], b.shape()[0], a.shape()[0],
      1.0, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], 0.0,
      c.begin(), a.shape()[1]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
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
void blas_ops::matmul_transab(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans,
      a.shape()[1], b.shape()[0], a.shape()[0],
      1.0, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], 0.0,
      c.begin(), a.shape()[1]);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
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

template<>
void blas_ops::matvec(const multi_array<double,2>& a, const multi_array<double,1>& b, multi_array<double,1>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_dgemv(CblasColMajor, CblasNoTrans,
      a.shape()[0], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), 1, 0.0,
      c.begin(), 1);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      double alpha = 1.0;
      double beta = 0.0;

      cublasDgemv(handle,CUBLAS_OP_N,a.shape()[0], a.shape()[1],
      &alpha, a.begin(), a.shape()[0],
      b.begin(), 1, &beta,
      c.begin(), 1);

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
void blas_ops::matvec_trans(const multi_array<double,2>& a, const multi_array<double,1>& b, multi_array<double,1>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_dgemv(CblasColMajor, CblasTrans,
      a.shape()[0], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), 1, 0.0,
      c.begin(), 1);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      double alpha = 1.0;
      double beta = 0.0;

      cublasDgemv(handle,CUBLAS_OP_T,a.shape()[0], a.shape()[1],
      &alpha, a.begin(), a.shape()[0],
      b.begin(), 1, &beta,
      c.begin(), 1);

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
void blas_ops::matvec_trans(const multi_array<float,2>& a, const multi_array<float,1>& b, multi_array<float,1>& c) const {
  if((a.sl == stloc::host) && (b.sl == stloc::host) && (c.sl == stloc::host)){ // everything on CPU
    cblas_sgemv(CblasColMajor, CblasTrans,
      a.shape()[0], a.shape()[1],
      1.0, a.begin(), a.shape()[0],
      b.begin(), 1, 0.0,
      c.begin(), 1);
    } else if ((a.sl == stloc::device) && (b.sl == stloc::device) && (c.sl == stloc::device)){ //everything on GPU
      #ifdef __CUDACC__
      float alpha = 1.0;
      float beta = 0.0;

      cublasSgemv(handle,CUBLAS_OP_T,a.shape()[0], a.shape()[1],
      &alpha, a.begin(), a.shape()[0],
      b.begin(), 1, &beta,
      c.begin(), 1);

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

diagonalization::diagonalization(Index m) {
  #ifdef __MKL__
  MKL_INT value = 0;
  char jobvs = 'V';
  char sort = 'N';
  MKL_INT nn   = m;
  MKL_INT lda  = m;
  MKL_INT ldvs = m;
  MKL_INT info;
  double work_opt;
  multi_array<double,1> diag_i({nn});
  multi_array<double,1> diag_r({nn});
  multi_array<double,2> D({m,m});
  multi_array<double,2> TT({m,m});
  #else
  int value = 0;
  char jobvs = 'V';
  char sort = 'N';
  int nn   = m;
  int lda  = m;
  int ldvs = m;
  int info;
  double work_opt;
  multi_array<double,1> diag_i({nn});
  multi_array<double,1> diag_r({nn});
  multi_array<double,2> D({m,m});
  multi_array<double,2> TT({m,m});
  #endif

  lwork = -1;
  dgees_(&jobvs,&sort,nullptr,&nn,D.begin(),&lda,&value,diag_r.begin(),diag_i.begin(),TT.begin(),&ldvs,&work_opt,&lwork,nullptr,&info);
  lwork = int(work_opt);
}

void diagonalization::operator()(const multi_array<double,2>& CC, multi_array<double,2>& TT, multi_array<double,1>& diag_r){
  #ifdef __MKL__
  MKL_INT value = 0;
  char jobvs = 'V';
  char sort = 'N';
  MKL_INT nn = CC.shape()[0];
  MKL_INT lda = CC.shape()[0];
  MKL_INT ldvs = CC.shape()[0];
  MKL_INT info;
  double work_opt;
  multi_array<double,1> diag_i({nn});
  multi_array<double,2> D(CC);
  #else
  int value = 0;
  char jobvs = 'V';
  char sort = 'N';
  int nn = CC.shape()[0];
  int lda = CC.shape()[0];
  int ldvs = CC.shape()[0];
  int info;
  double work_opt;
  multi_array<double,1> diag_i({nn});
  multi_array<double,2> D(CC);
  #endif

  if(lwork == -1){ // Dumb call to obtain optimal value to work
  dgees_(&jobvs,&sort,nullptr,&nn,D.begin(),&lda,&value,diag_r.begin(),diag_i.begin(),TT.begin(),&ldvs,&work_opt,&lwork,nullptr,&info);
    lwork = int(work_opt);
  }else{
    multi_array<double,1> work({lwork});
    dgees_(&jobvs,&sort,nullptr,&nn,D.begin(),&lda,&value,diag_r.begin(),diag_i.begin(),TT.begin(),&ldvs,work.begin(),&lwork,nullptr,&info);
  }
}
