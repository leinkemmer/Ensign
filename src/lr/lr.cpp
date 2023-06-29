#include <generic/kernels.hpp>
#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/timer.hpp>
#include <lr/coefficients.hpp>

#include <cstring>

#ifdef __CUDACC__
#include <curand.h>
#endif

template<class T>
std::function<T(T*,T*)> inner_product_from_weight(T* w, Index N) {
  return [w,N](T* a, T*b) {
    T result=T(0.0);
    #ifdef __OPENMP__
    #pragma omp parallel for reduction(+:result)
    #endif
    for(Index i=0;i<N;i++)
    result += w[i]*a[i]*b[i];
    return result;
  };
};
template std::function<double(double*,double*)> inner_product_from_weight(double* w, Index N);
template std::function<float(float*,float*)> inner_product_from_weight(float* w, Index N);


template<>
std::function<double(double*,double*)> inner_product_from_const_weight(double w, Index N) {
  return [w,N](double* a, double*b) {
    double result = cblas_ddot(N, a, 1, b, 1);
    result *= w;
    return result;
  };
};

template<>
std::function<float(float*,float*)> inner_product_from_const_weight(float w, Index N) {
  return [w,N](float* a, float*b) {
    float result = cblas_sdot(N, a, 1, b, 1);
    result *= w;
    return result;
  };
};

template std::function<double(double*,double*)> inner_product_from_const_weight(double w, Index N);
template std::function<float(float*,float*)> inner_product_from_const_weight(float w, Index N);



void gram_schmidt_cpu(multi_array<double,2>& Q, multi_array<double,2>& R, std::function<double(double*,double*)> inner_product) {
  array<Index,2> dims = Q.shape();

  std::default_random_engine generator(1234);
  std::normal_distribution<double> distribution(0.0,1.0);

  for(Index j=0;j<dims[1];j++) {
    for(Index k=0;k<j;k++) {
      R(k,j) = inner_product(Q.extract({j}), Q.extract({k}));
      cblas_daxpy(dims[0], -R(k,j), Q.extract({k}), 1, Q.extract({j}),1);
      R(j,k) = 0.0;
    }
    double ip = inner_product(Q.extract({j}), Q.extract({j}));
    R(j,j) = sqrt(ip);

    if(R(j,j) > 1e-14){
      cblas_dscal(dims[0],1.0/R(j,j),Q.extract({j}),1);
    } else {

      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index l = 0; l < dims[0]; l++){
        Q(l,j) = distribution(generator);
      }

      for(Index k=0;k<j;k++) {
        double val = inner_product(Q.extract({j}), Q.extract({k}));
        cblas_daxpy(dims[0], -val, Q.extract({k}), 1, Q.extract({j}),1);
      }

      double nrm = sqrt(inner_product(Q.extract({j}), Q.extract({j})));
      cblas_dscal(dims[0],1.0/nrm,Q.extract({j}),1);

    }
  }
};

#ifdef __CUDACC__
void gram_schmidt_gpu(multi_array<double,2>& Q, multi_array<double,2>& R, double w, curandGenerator_t gen, cublasHandle_t handle_devres) { //with constant weight for inner product
  Index n = Q.shape()[0];
  int r = Q.shape()[1];
  double* nrm;
  cudaMalloc((void**)&nrm,sizeof(double));

  for(Index j=0;j<r;j++) {
    for(Index k=0;k<j;k++) {
      cublasDdot (handle_devres, n, &Q(0,j), 1, &Q(0,k), 1, &R(k,j));
      cudaDeviceSynchronize();
      scale_unique<<<1,1>>>(&R(k,j),w); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
      dmaxpy<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &R(k,j), &Q(0,k), &Q(0,j));
      scale_unique<<<1,1>>>(&R(j,k),0.0);
    }

      cublasDdot (handle_devres, n, &Q(0,j), 1, &Q(0,j), 1, &R(j,j));
      cudaDeviceSynchronize();
      scale_sqrt_unique<<<1,1>>>(&R(j,j),w);

      double val;
      cudaMemcpy(&val,&R(j,j),sizeof(double),cudaMemcpyDeviceToHost);

      if(std::abs(val) > 1e-14){
        ptw_div_gs<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &Q(0,j), &R(j,j));
      } else{

        // Generate random
        curandGenerateNormalDouble(gen, &Q(0,j), n, 0.0, 1.0);

        for(Index k=0;k<j;k++) {
          cublasDdot (handle_devres, n, &Q(0,j), 1, &Q(0,k), 1, nrm);
          cudaDeviceSynchronize();
          scale_unique<<<1,1>>>(nrm,w); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
          dmaxpy<<<(n+n_threads-1)/n_threads,n_threads>>>(n, nrm, &Q(0,k), &Q(0,j));
        }

          cublasDdot (handle_devres, n, &Q(0,j), 1, &Q(0,j), 1, nrm);
          cudaDeviceSynchronize();
          scale_sqrt_unique<<<1,1>>>(nrm,w);
          ptw_div_gs<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &Q(0,j), nrm);

      }

  }
};

// STILL TO BE TESTED
/*
void gram_schmidt_gpu(multi_array<double,2>& Q, multi_array<double,2>& R, double* w) { //with non-constant weight for inner product. Still to be tested

  Index n = Q.shape()[0];
  int r = Q.shape()[1];
  multi_array<double,1> tmp({n},stloc::device);

  for(Index j=0;j<r;j++) {
    for(Index k=0;k<j;k++) {
      ptw_mult<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &Q(0,j),w,tmp.begin());
      cublasDdot (handle_devres, n, tmp.begin(), 1, &Q(0,k), 1,&R(k,j));
      cudaDeviceSynchronize();
      dmaxpy<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &R(k,j), &Q(0,k), &Q(0,j));
      scale_unique<<<1,1>>>(&R(j,k),0.0);
    }
      ptw_mult<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &Q(0,j),w,tmp.begin());
      cublasDdot (handle_devres, n, &Q(0,j), 1, tmp.begin(), 1,&R(j,j));
      cudaDeviceSynchronize();
      ptw_div_gs<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &Q(0,j), &R(j,j));

  }
};
*/

#endif


gram_schmidt::gram_schmidt(const blas_ops* _blas) {
  blas = _blas;

  #ifdef __CUDACC__
  gen = 0;
  if(blas->gpu) {
    curandStatus_t status = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    if(status != CURAND_STATUS_SUCCESS) {
        cout << "ERROR: curandCreateGenerator failsed. Error code: " << status <<  endl;
        exit(1);
    }
    curandSetPseudoRandomGeneratorSeed(gen,1234);
  }
  #endif
}

gram_schmidt::~gram_schmidt() {
  #ifdef __CUDACC__
  if(gen)
      curandDestroyGenerator(gen);
  #endif
}

void gram_schmidt::operator()(multi_array<double,2>& Q, multi_array<double,2>& R, std::function<double(double*,double*)> inner_product) {
  if(Q.sl == stloc::host) {
    gram_schmidt_cpu(Q, R, inner_product);
  } else {
    cout << "ERROR: gram_schmidt::operator() with non-constant inner product currently not implemented for GPU." << endl;
    exit(1);
  }
}

void gram_schmidt::operator()(multi_array<double,2>& Q, multi_array<double,2>& R, double w) {
  if(Q.sl == stloc::host) {
    cout << "ERROR: gram_schmidt::operator() with constant inner product currently not implemented on CPU." << endl;
    exit(1);
  } else {
    #ifdef __CUDACC__
    gram_schmidt_gpu(Q, R, w, gen, blas->handle_devres);
    #else
    cout << "ERROR: gram_schmidt_gpu called but no GPU support available." << endl;
    exit(1);
    #endif
  }
}




/*
template<>
void gram_schmidt(multi_array<float,2>& Q, multi_array<float,2>& R, std::function<float(float*,float*)> inner_product) {
  array<Index,2> dims = Q.shape();
  for(Index j=0;j<dims[1];j++) {
    for(Index k=0;k<j;k++) {
      R(k,j) = inner_product(Q.extract({j}), Q.extract({k}));
      cblas_saxpy(dims[0], -R(k,j), Q.extract({k}), 1, Q.extract({j}),1);
      R(j,k) = float(0.0);
    }
    R(j,j) = sqrt(inner_product(Q.extract({j}), Q.extract({j})));
    if(std::abs(R(j,j)) < float(1000)*std::numeric_limits<float>::epsilon()){
      cout << "Warning: linearly dependent columns in Gram-Schmidt" << endl;
    } else{
      cblas_sscal(dims[0],float(1.0/R(j,j)),Q.extract({j}),1);
    }
  }
};*/


template<class T>
void initialize(lr2<T>& lr, vector<const T*> X, vector<const T*> V, std::function<T(T*,T*)> inner_product_X, std::function<T(T*,T*)> inner_product_V, const blas_ops& blas) {

  int n_b = X.size();
  Index r = lr.rank();

  std::default_random_engine generator(1234);
  std::normal_distribution<double> distribution(0.0,1.0);

  for(Index k=0;k<r;k++) {
    if(k < n_b){
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0;i<lr.size_X();i++) {
        lr.X(i, k) = X[k][i];
      }
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0;i<lr.size_V();i++) {
        lr.V(i, k) = V[k][i];
      }
    }
    else{
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0;i<lr.size_X();i++) {
        lr.X(i, k) = distribution(generator);
      }
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0;i<lr.size_V();i++) {
        lr.V(i, k) = distribution(generator);
      }
    }
    }

  multi_array<T, 2> X_R(lr.S.shape()), V_R(lr.S.shape());

  gram_schmidt gs(&blas);
  gs(lr.X, X_R, inner_product_X);
  gs(lr.V, V_R, inner_product_V);

  for(int j = n_b; j < r; j++){
    for(int i = 0; i < r; i++){
      X_R(i,j) = T(0.0);
    }
  }

  for(int j = n_b; j < r; j++){
    for(int i = 0; i < r; i++){
      V_R(i,j) = T(0.0);
    }
  }

  blas.matmul_transb(X_R, V_R, lr.S);

};
template void initialize(lr2<double>& lr, vector<const double*> X, vector<const double*> V, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V, const blas_ops& blas);
//template void initialize(lr2<float>& lr, vector<const float*> X, vector<const float*> V, std::function<float(float*,float*)> inner_product_X, std::function<float(float*,float*)> inner_product_V, const blas_ops& blas);




template<class T>
void lr_add(vector<const lr2<T>*> A, const vector<T>& alpha, lr2<T>& out,
            std::function<T(T*,T*)> inner_product_X,
            std::function<T(T*,T*)> inner_product_V,
            const blas_ops& blas) {

  // check dimensions
  Index total_r=0;
  for(size_t k=0;k<A.size();k++) {
    total_r += A[k]->rank();
    if(A[k]->size_X() != out.size_X()) {
      cout << "ERROR in lr_add: shape of X does not match in input " << k << " and output." << endl;
      exit(1);
    }
    if(A[k]->size_V() != out.size_V()) {
      cout << "ERROR in lr_add: shape of V does not match in input " << k << " and output." << endl;
      exit(1);
    }
  }
  if(total_r != out.rank()) {
    cout << "ERROR in lr_add: output rank does not match sum of input ranks." << endl;
    exit(1);
  }

  gram_schmidt gs(&blas);

  // copy elements of X in out.V and orthogonalize
  {
    Index offset = 0;
    for(Index k=0;k<(Index)A.size();k++) {
      Index size = A[k]->size_X()*A[k]->rank();
      std::memcpy(out.X.data()+offset, A[k]->X.data(), sizeof(T)*size);
      offset += size;
    }
  }

  multi_array<T, 2> R_X({out.rank(), out.rank()});
  gs(out.X, R_X, inner_product_X);

  // copy elements of V in out.V and orthogonalize
  {
    Index offset = 0;
    for(Index k=0;k<(Index)A.size();k++) {
      Index size = A[k]->size_V()*A[k]->rank();
      std::memcpy(out.V.data()+offset, A[k]->V.data(), sizeof(T)*size);
      offset += size;
    }
  }

  multi_array<T, 2> R_V({out.rank(), out.rank()});
  gs(out.V, R_V, inner_product_V);

  // copy elements of S in out.S
  {
    std::fill(out.S.begin(), out.S.end(), 0.0);
    Index offset = 0;
    for(Index k=0;k<(Index)A.size();k++) {
      Index r_k = A[k]->rank();

      for(Index j=0;j<r_k;j++)
        for(Index i=0;i<r_k;i++)
          out.S(i + offset, j + offset) = alpha[k]*A[k]->S(i, j);

      offset += r_k;
    }
  }

  multi_array<T,2> tmp = out.S;
  blas.matmul(R_X, tmp, out.S);

  tmp = out.S;
  blas.matmul_transb(tmp, R_V, out.S);
}


template<class T>
void lr_add(T alpha, const lr2<T>& A, T beta, const lr2<T>& B, lr2<T>& out,
            std::function<T(T*,T*)> inner_product_X,
            std::function<T(T*,T*)> inner_product_V,
            const blas_ops& blas) {

  lr_add({&A, &B}, {alpha, beta}, out, inner_product_X, inner_product_V, blas);
}

template void lr_add(double alpha, const lr2<double>& A, double beta, const lr2<double>& B, lr2<double>& out, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V, const blas_ops& blas);


template<class T>
void lr_add(T alpha, const lr2<T>& A, T beta, const lr2<T>& B,
            T gamma, const lr2<T>& C, lr2<T>& out,
            std::function<T(T*,T*)> inner_product_X,
            std::function<T(T*,T*)> inner_product_V,
            const blas_ops& blas) {

  lr_add({&A, &B, &C}, {alpha, beta, gamma}, out, inner_product_X, inner_product_V, blas);
}

  
template void lr_add(double alpha, const lr2<double>& A, double beta, const lr2<double>& B, double gamma, const lr2<double>& C, lr2<double>& out, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V, const blas_ops& blas);


template<class T>
void lr_mul(const lr2<T>& A, const lr2<T>& B, lr2<T>& out,
            std::function<T(T*,T*)> inner_product_X,
            std::function<T(T*,T*)> inner_product_V,
            const blas_ops& blas) {

  // check dimensions
  if(A.size_X() != out.size_X()) {
    cout << "ERROR in lr_mul: shape of X does not match in input A and output." << endl;
    exit(1);
  }
  if(A.size_V() != out.size_V()) {
    cout << "ERROR in lr_mul: shape of V does not match in input B and output." << endl;
    exit(1);
  }
  Index r_A = A.rank();
  Index r_B = B.rank();
  if(r_A*r_B != out.rank()) {
    cout << "ERROR in lr_mul: output rank does not match sum of input ranks." << endl;
    exit(1);
  }

  gram_schmidt gs(&blas);

  // construct the new X basis and orthogonalize
  for(Index k=0;k<r_B;k++)
    for(Index i=0;i<r_A;i++)
      for(Index n=0;n<A.size_X();n++)
        out.X(n, i+r_A*k) = A.X(n,i)*B.X(n,k);

  multi_array<T, 2> R_X({out.rank(), out.rank()});
  gs(out.X, R_X, inner_product_X);

  // construct the new V basis and orthogonalize
  for(Index l=0;l<r_B;l++)
      for(Index j=0;j<r_A;j++)
        for(Index n=0;n<A.size_V();n++)
          out.V(n, j+r_A*l) = A.V(n,j)*B.V(n,l);
    
    multi_array<T, 2> R_V({out.rank(), out.rank()});
    gs(out.V, R_V, inner_product_V);

    // construct S
  for(Index k=0;k<r_B;k++)
    for(Index i=0;i<r_A;i++)
      for(Index l=0;l<r_B;l++)
        for(Index j=0;j<r_A;j++)
          out.S(i+r_A*k, j+r_A*l) = A.S(i,j)*B.S(k,l);

  multi_array<T,2> tmp = out.S;
  blas.matmul(R_X, tmp, out.S);
    
  tmp = out.S;
  blas.matmul_transb(tmp, R_V, out.S);
}

template void lr_mul(const lr2<double>& A, const lr2<double>& B, lr2<double>& out, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V, const blas_ops& blas);


template<class T>
void lr_truncate(const lr2<T>& in, lr2<T>& out, const blas_ops& blas) {
  if(out.rank() > in.rank()) {
    cout << "ERROR in lr_truncate: rank of output is larger than rank of input." << endl;
    exit(1);
  }

  // compute the SVD
  multi_array<T, 2> U({in.rank(), in.rank()}); 
  multi_array<T, 2> V({in.rank(), in.rank()}); 
  multi_array<T, 1> sv({in.rank()});
  svd(in.S, U, V, sv, blas);

  // Here we use that for blas.matmul the rows of the output can be smaller than
  // the rows of the input. Only the desired partial result is then computed.
  blas.matmul(in.X, U, out.X);
  blas.matmul(in.V, V, out.V);

  for(Index j=0;j<out.rank();j++)
    for(Index i=0;i<out.rank();i++)
      out.S(i, j) = (i==j) ? sv(i) : 0.0;

}

template void lr_truncate(const lr2<double>& in, lr2<double>& out, const blas_ops& blas);


template<class T>
double lr_inner_product(const lr2<T>& A, const lr2<T>&B, T w, const blas_ops& blas) {
  multi_array<double,2> XtX({A.rank(), B.rank()});
  blas.matmul_transa(A.X, B.X, XtX);
  
  multi_array<double,2> VtV({A.rank(), B.rank()});
  blas.matmul_transa(A.V, B.V, VtV);

  multi_array<double,2> C({A.rank(), B.rank()}), D({A.rank(), B.rank()});
  blas.matmul_transa(A.S, XtX, C);
  blas.matmul_transb(VtV, B.S, D);

  double ip = 0.0;
  for(Index j=0;j<B.rank();j++)
    for(Index i=0;i<A.rank();i++)
      ip += w*C(i,j)*D(i,j);
  return ip;
}

template double lr_inner_product(const lr2<double>& A, const lr2<double>&B, double w, const blas_ops& blas);


template<class T>
double lr_norm_sq(const lr2<T>& A, const blas_ops& blas) {
  double norm_sq = 0.0;
  for(Index j=0;j<A.rank();j++)
    for(Index i=0;i<A.rank();i++)
      norm_sq += pow(A.S(i,j), 2);
  return norm_sq;
}

template double lr_norm_sq(const lr2<double>& A, const blas_ops& blas);
