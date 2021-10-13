#include <generic/matrix.hpp>
#include <generic/timer.hpp>

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

template<class T>
void set_const(multi_array<T,1>& a,T alpha) {
  if(a.sl == stloc::host) {
    fill(a.begin(), a.end(), alpha);
  } else {
    #ifdef __CUDACC__
      fill_gpu<<<(a.shape()[0]+n_threads-1)/n_threads,n_threads>>>(a.shape()[0],a.begin(),alpha);
    #else
    cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
    << __LINE__ << endl;
    exit(1);
    #endif
  }
}
template void set_const(multi_array<double,1>&, double);
template void set_const(multi_array<float,1>&, float);

template<class T>
void set_const2(multi_array<T,2>& a,T alpha) {
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
template void set_const2(multi_array<double,2>&, double);
template void set_const2(multi_array<float,2>&, float);

template<class T>
void ptw_mult_row(const multi_array<T,2>& a, const T* w, multi_array<T,2>& out){
  if(a.sl == stloc::host){
    Index N = a.shape()[0];
    #ifdef __OPENMP__
    #pragma omp parallel for collapse(2)
    for(int r = 0; r < a.shape()[1]; r++){
      for(Index i = 0; i < a.shape()[0]; i++){
        const T* ptr = a.extract({r});
        out(i,r) = ptr[i]*w[i];
      }
    }
    #else
    for(int r = 0; r < a.shape()[1]; r++){
      T* ptr = a.extract({r});
      for(Index i = 0; i < a.shape()[0]; i++){
        out(i,r) = ptr[i]*w[i];
      }
    }
    #endif
  }else{
    #ifdef __CUDACC__
    //ptw_mult_row_k<<<2,2>>>(a.num_elements(), a.shape()[1], a.begin(), w); TO DO MANAGE COMPLEX NUMBERS
    #else
    cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
    << __LINE__ << endl;
    exit(1);
    #endif

  }

}
template void ptw_mult_row(const multi_array<double,2>&, const double*, multi_array<double,2>&);
template void ptw_mult_row(const multi_array<float,2>&, const float*, multi_array<float,2>&);
template void ptw_mult_row(const multi_array<complex<double>,2>&, const complex<double>*, multi_array<complex<double>,2>&);
template void ptw_mult_row(const multi_array<complex<float>,2>&, const complex<float>*, multi_array<complex<float>,2>&);


template<class T> // kernel?
void transpose_inplace(multi_array<T,2>& a){
  Index m = a.shape()[0];
  T tmp = 0.0;
  for(Index r = 0; r < m; r++){
    for(Index i = 1 + r; i < m; i++){
      tmp = a(r,i);
      a(r,i) = a(i,r);
      a(i,r) = tmp;
    }
  }
}

template void transpose_inplace(multi_array<double,2>&);
template void transpose_inplace(multi_array<float,2>&);

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
    complex<double> one(1.0,0.0);
    complex<double> zero(0.0,0.0);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      a.shape()[0], b.shape()[1], a.shape()[1],
      &one, a.begin(), a.shape()[0],
      b.begin(), a.shape()[1], &zero,
      c.begin(), a.shape()[0]);

}

#ifdef __CUDACC__
template<>
void matmul(const multi_array<cuDoubleComplex,2>& a, const multi_array<cuDoubleComplex,2>& b, multi_array<cuDoubleComplex,2>& c){

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
void matmul(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c){
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
    complex<double> one(1.0,0.0);
    complex<double> zero(0.0,0.0);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      a.shape()[0], b.shape()[0], a.shape()[1],
      &one, a.begin(), a.shape()[0],
      b.begin(), b.shape()[0], &zero,
      c.begin(), a.shape()[0]);
}

#ifdef __CUDACC__
template<>
void matmul_transb(const multi_array<cuDoubleComplex,2>& a, const multi_array<cuDoubleComplex,2>& b, multi_array<cuDoubleComplex,2>& c){
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
void matmul_transab(const multi_array<double,2>& a, const multi_array<double,2>& b, multi_array<double,2>& c){
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
void matmul_transab(const multi_array<float,2>& a, const multi_array<float,2>& b, multi_array<float,2>& c){
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
void matvec(const multi_array<double,2>& a, const multi_array<double,1>& b, multi_array<double,1>& c){
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
void matvec_trans(const multi_array<double,2>& a, const multi_array<double,1>& b, multi_array<double,1>& c){
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
void matvec_trans(const multi_array<float,2>& a, const multi_array<float,1>& b, multi_array<float,1>& c){
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

array<fftw_plan,2> create_plans_1d(Index dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq){
  array<fftw_plan,2> out;
  int dims = int(dims_);

  out[0] = fftw_plan_many_dft_r2c(1, &dims, real.shape()[1], real.begin(), NULL, 1, dims, (fftw_complex*)freq.begin(), NULL, 1, dims/2 + 1, FFTW_MEASURE);
  out[1] = fftw_plan_many_dft_c2r(1, &dims, real.shape()[1], (fftw_complex*)freq.begin(), NULL, 1, dims/2 + 1, real.begin(), NULL, 1, dims, FFTW_MEASURE);

  return out;
}

array<fftw_plan,2> create_plans_1d(Index dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq){
  array<fftw_plan,2> out;
  int dims = int(dims_);

  out[0] = fftw_plan_many_dft_r2c(1, &dims, 1, real.begin(), NULL, 1, dims, (fftw_complex*)freq.begin(), NULL, 1, dims/2 + 1, FFTW_MEASURE);
  out[1] = fftw_plan_many_dft_c2r(1, &dims, 1, (fftw_complex*)freq.begin(), NULL, 1, dims/2 + 1, real.begin(), NULL, 1, dims, FFTW_MEASURE);

  return out;
}

array<fftw_plan,2> create_plans_2d(array<Index,2> dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq){
  array<fftw_plan,2> out;
  array<int,2> dims = {int(dims_[1]),int(dims_[0])};

  out[0] = fftw_plan_many_dft_r2c(2, dims.begin(), real.shape()[1], real.begin(), NULL, 1, dims[1]*dims[0], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), FFTW_MEASURE);
  out[1] = fftw_plan_many_dft_c2r(2, dims.begin(), real.shape()[1], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), real.begin(), NULL, 1, dims[1]*dims[0], FFTW_MEASURE);

  return out;
}

array<fftw_plan,2> create_plans_2d(array<Index,2> dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq){
  array<fftw_plan,2> out;
  array<int,2> dims = {int(dims_[1]),int(dims_[0])};

  out[0] = fftw_plan_many_dft_r2c(2, dims.begin(), 1, real.begin(), NULL, 1, dims[1]*dims[0], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), FFTW_MEASURE);
  out[1] = fftw_plan_many_dft_c2r(2, dims.begin(), 1, (fftw_complex*)freq.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), real.begin(), NULL, 1, dims[1]*dims[0], FFTW_MEASURE);

  return out;
}

array<fftw_plan,2> create_plans_3d(array<Index,3> dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq){
  array<fftw_plan,2> out;
  array<int,3> dims = {int(dims_[2]), int(dims_[1]), int(dims_[0])};

  out[0] = fftw_plan_many_dft_r2c(3, dims.begin(), real.shape()[1], real.begin(), NULL, 1, dims[2]*dims[1]*dims[0], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), FFTW_MEASURE);
  out[1] = fftw_plan_many_dft_c2r(3, dims.begin(), real.shape()[1], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), real.begin(), NULL, 1, dims[2]*dims[1]*dims[0], FFTW_MEASURE);

  return out;
}

array<fftw_plan,2> create_plans_3d(array<Index,3> dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq){
  array<fftw_plan,2> out;
  array<int,3> dims = {int(dims_[2]), int(dims_[1]), int(dims_[0])};

  out[0] = fftw_plan_many_dft_r2c(3, dims.begin(), 1, real.begin(), NULL, 1, dims[2]*dims[1]*dims[0], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), FFTW_MEASURE);
  out[1] = fftw_plan_many_dft_c2r(3, dims.begin(), 1, (fftw_complex*)freq.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), real.begin(), NULL, 1, dims[2]*dims[1]*dims[0], FFTW_MEASURE);

  return out;
}

void destroy_plans(array<fftw_plan,2>& plans){
  fftw_destroy_plan(plans[0]);
  fftw_destroy_plan(plans[1]);
}

#ifdef __MKL__
  void schur(multi_array<double,2>& CC, multi_array<double,2>& TT, multi_array<double,1>& diag_r, MKL_INT& lwork){
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

    if(lwork == -1){ // Dumb call to obtain optimal value to work
    dgees_(&jobvs,&sort,nullptr,&nn,D.begin(),&lda,&value,diag_r.begin(),diag_i.begin(),TT.begin(),&ldvs,&work_opt,&lwork,nullptr,&info);
      lwork = int(work_opt);
    }else{
      multi_array<double,1> work({lwork});
      dgees_(&jobvs,&sort,nullptr,&nn,D.begin(),&lda,&value,diag_r.begin(),diag_i.begin(),TT.begin(),&ldvs,work.begin(),&lwork,nullptr,&info);
    }
  }
#else
void schur(multi_array<double,2>& CC, multi_array<double,2>& TT, multi_array<double,1>& diag_r, int& lwork){
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

  if(lwork == -1){ // Dumb call to obtain optimal value to work
    dgees_(&jobvs,&sort,nullptr,&nn,D.begin(),&lda,&value,diag_r.begin(),diag_i.begin(),TT.begin(),&ldvs,&work_opt,&lwork,nullptr,&info);
    lwork = int(work_opt);
  }else{
    multi_array<double,1> work({lwork});
    dgees_(&jobvs,&sort,nullptr,&nn,D.begin(),&lda,&value,diag_r.begin(),diag_i.begin(),TT.begin(),&ldvs,work.begin(),&lwork,nullptr,&info);
  }
}
#endif

#ifdef __CUDACC__
/*
  array<cufftHandle,2> create_plans_1d(Index dims_){
    array<cufftHandle,2> out;
    int dims = int(dims_);
    cufftPlan1d(&out[0], dims, CUFFT_D2Z,1);
    cufftPlan1d(&out[1], dims, CUFFT_Z2D,1);

    return out;
  }
*/
  array<cufftHandle,2> create_plans_1d(Index dims_, int howmany){
    array<cufftHandle,2> out;
    int dims = int(dims_);

    cufftPlanMany(&out[0], 1, &dims, NULL, 1, dims, NULL, 1, dims/2 + 1, CUFFT_D2Z, howmany);
    cufftPlanMany(&out[1], 1, &dims, NULL, 1, dims/2 + 1, NULL, 1, dims, CUFFT_Z2D, howmany);

    return out;
  }
/*
  array<cufftHandle,2> create_plans_2d(array<Index,2> dims_){
    array<cufftHandle,2> out;

    cufftPlan2d(&out[0],int(dims_[1]),int(dims_[0]),CUFFT_D2Z);
    cufftPlan2d(&out[1],int(dims_[1]),int(dims_[0]),CUFFT_Z2D);

    return out;
  }
*/
  array<cufftHandle,2> create_plans_2d(array<Index,2> dims_, int howmany){
    array<cufftHandle,2> out;
    array<int,2> dims = {int(dims_[1]),int(dims_[0])};

    cufftPlanMany(&out[0], 2, dims.begin(), NULL, 1, dims[1]*dims[0], NULL, 1, dims[0]*(dims[1]/2 + 1), CUFFT_D2Z, howmany);
    cufftPlanMany(&out[1], 2, dims.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), NULL, 1, dims[1]*dims[0], CUFFT_Z2D, howmany);

    return out;
  }
/*
  array<cufftHandle,2> create_plans_3d(array<Index,3> dims_){
    array<cufftHandle,2> out;

    cufftPlan3d(&out[0],int(dims_[2]),int(dims_[1]),int(dims_[0]),CUFFT_D2Z);
    cufftPlan3d(&out[1],int(dims_[2]),int(dims_[1]),int(dims_[0]),CUFFT_Z2D);

    return out;
  }
*/
  array<cufftHandle,2> create_plans_3d(array<Index,3> dims_, int howmany){
    array<cufftHandle,2> out;
    array<int,3> dims = {int(dims_[2]),int(dims_[1]),int(dims_[0])};

    cufftPlanMany(&out[0], 3, dims.begin(), NULL, 1, dims[2]*dims[1]*dims[0], NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), CUFFT_D2Z, howmany);
    cufftPlanMany(&out[1], 3, dims.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), NULL, 1, dims[2]*dims[1]*dims[0], CUFFT_Z2D, howmany);

    return out;
  }

  void destroy_plans(array<cufftHandle,2>& plans){
    cufftDestroy(plans[0]);
    cufftDestroy(plans[1]);
  }

#endif

//template<class T>
//void matmul_transa(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

//template<class T>
//void matmul_transb(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);


//template struct multi_array<double,2>;
//template struct multi_array<float,2>;

//template void set_zero(multi_array<double,2>&);
//template void set_zero(multi_array<float,2>&);
