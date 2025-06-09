#include <generic/kernels.hpp>
#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/timer.hpp>
#include <lr/coefficients.hpp>

#include <cstring>
#include <Eigen/Dense>

#ifdef __CUDACC__
#include <curand.h>
#endif

namespace Ensign {

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
    //double result=0.0;
    //for(Index i=0;i<N;i++){
    //  result += a[i]*b[i];
    //}
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

void orthogonalize_householder_constw(multi_array<double,2>& Q, multi_array<double,2>& R, double w) { //Removed blas argument because not needed
  array<Index,2> dims = Q.shape();

  using namespace Eigen;
  MatrixXd A(dims[0], dims[1]);
  for(Index j=0;j<dims[1];j++) {
    for(Index i=0;i<dims[0];i++) {
      A(i,j) = Q(i,j);
    }
  }

  HouseholderQR<MatrixXd> qr(A.rows(), A.cols());
  qr.compute(A);
  MatrixXd q = qr.householderQ()*MatrixXd::Identity(A.rows(), A.cols());
  MatrixXd temp = qr.matrixQR().triangularView<Upper>();

  MatrixXd RR(A.cols(), A.cols());
  RR.setZero();
  for(Index j=0;j<std::min(A.cols(),temp.cols());j++)
    for(Index i=0;i<std::min(A.cols(),temp.rows());i++)
      RR(i,j) = temp(i,j);

  for(Index j=0;j<dims[1];j++) {
    for(Index i=0;i<dims[0];i++) {
      Q(i,j) = q(i,j);
    }
  }

  for(Index j=0;j<dims[1];j++)
    for(Index i=0;i<dims[0];i++)
      Q(i,j) /= sqrt(w);

  for(Index j=0;j<dims[1];j++)
    for(Index i=0;i<dims[1];i++)
      R(i,j) = sqrt(w)*RR(i,j);

}


void orthogonalize_householder_vecw(multi_array<double,2>& Q, multi_array<double,2>& R, double* w) { //Removed blas argument because not needed
  array<Index,2> dims = Q.shape();

  // We first multiply our matrix with the diagonal matrix diag(w^1/2) from the left

  for(Index j=0;j<dims[1];j++)
    for(Index i=0;i<dims[0];i++)
      Q(i,j) *= sqrt(w[i]);


  // Now we rewrite our matrix and do QR decomposition

  using namespace Eigen;
  MatrixXd A(dims[0], dims[1]);
  for(Index j=0;j<dims[1];j++) {
    for(Index i=0;i<dims[0];i++) {
      A(i,j) = Q(i,j);
    }
  }

  HouseholderQR<MatrixXd> qr(A.rows(), A.cols());
  qr.compute(A);
  MatrixXd q = qr.householderQ()*MatrixXd::Identity(A.rows(), A.cols());
  MatrixXd temp = qr.matrixQR().triangularView<Upper>();


  MatrixXd RR(A.cols(), A.cols());
  RR.setZero();
  for(Index j=0;j<std::min(A.cols(),temp.cols());j++)
    for(Index i=0;i<std::min(A.cols(),temp.rows());i++)
      RR(i,j) = temp(i,j);


  for(Index j=0;j<dims[1];j++) {
    for(Index i=0;i<dims[0];i++) {
      Q(i,j) = q(i,j);
    }
  }


  // Now we need to multiply our Q matrix with diag(w^-1/2) from the left
  // The RR matrix we need to convert to our format

  for(Index j=0;j<dims[1];j++)
    for(Index i=0;i<dims[0];i++)
      Q(i,j) /= sqrt(w[i]);

  for(Index j=0;j<dims[1];j++)
    for(Index i=0;i<dims[1];i++)
      R(i,j) = RR(i,j);

}


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


void orthogonalize_householder_constw_gpu(multi_array<double,2>& Q, multi_array<double,2>& R, double w, cusolverDnHandle_t handle_cusolver) {

  int m = Q.shape()[0];   // also lda
  int n = Q.shape()[1];
  

  // allocate memory for tau, work and device info
  double *devTau, *work;
	int szWork;

  cudaMalloc((void**)&devTau, n * sizeof(double));

  cusolverDnDgeqrf_bufferSize(handle_cusolver, m, n, &Q(0,0), m, &szWork);
  cudaDeviceSynchronize();

  cudaMalloc((void**)&work, szWork * sizeof(double));

  int *devInfo;
  cudaMalloc((void **)&devInfo, sizeof(int));

  // do QR factorization
  cusolverDnDgeqrf(handle_cusolver, m, n, &Q(0,0), m, devTau, work, szWork, devInfo); 
  cudaDeviceSynchronize();

  // copy data from Q to our multi_array R (only upper tridiagonal part of A is R, rest is 0) (R should already be nxn)
  // in our function we already multiply by sqrt(w)

  copy_R<<<n,n>>>(m, n, &Q(0,0), &R(0,0), w);

  // we don't allocate extra memory for our second usage of cuSolver, because szwork2 < szwork
  // calculate the orthogonal matrix Q
  cusolverDnDorgqr(handle_cusolver, m, n, n, &Q(0,0), m, devTau, work, szWork, devInfo);
  cudaDeviceSynchronize();

  // divide Q by sqrt(w)

  div_Q<<<m,n>>>(m, n, &Q(0,0), w);

}


#endif


orthogonalize::orthogonalize(const Ensign::Matrix::blas_ops* _blas) {
  blas = _blas;

  #ifdef __CUDACC__
  gen = 0;
  if(blas->gpu) {
    curandStatus_t status = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    if(status != CURAND_STATUS_SUCCESS) {
        cout << "ERROR: curandCreateGenerator failed. Error code: " << status <<  endl;
        exit(1);
    }
    curandSetPseudoRandomGeneratorSeed(gen,1234);
  }
  #endif
}

orthogonalize::~orthogonalize() {
  #ifdef __CUDACC__
  if(gen)
      curandDestroyGenerator(gen);
  #endif
}

void orthogonalize::operator()(multi_array<double,2>& Q, multi_array<double,2>& R, std::function<double(double*,double*)> inner_product) {
  if(Q.sl == stloc::host) {
    gram_schmidt_cpu(Q, R, inner_product);
  } else {
    cout << "ERROR: orthogonalize::operator() with non-constant inner product currently not implemented for GPU." << endl;
    exit(1);
  }
}

void orthogonalize::operator()(multi_array<double,2>& Q, multi_array<double,2>& R, double w) {
  if(Q.sl == stloc::host) {
    orthogonalize_householder_constw(Q, R, w);
  } else {
    #ifdef __CUDACC__
    //gram_schmidt_gpu(Q, R, w, gen, blas->handle_devres);
    orthogonalize_householder_constw_gpu(Q, R, w, blas->handle_cusolver);
    #else
    cout << "ERROR: orthogonalize_gpu called but no GPU support available." << endl;
    exit(1);
    #endif
  }
}


void orthogonalize::operator()(multi_array<double,2>& Q, multi_array<double,2>& R, double* w) {
  if(Q.sl == stloc::host) {
    orthogonalize_householder_vecw(Q, R, w);
  } else {
    cout << "ERROR: orthogonalize::operator() with constant vector inner product currently not implemented for GPU." << endl;
    exit(1);
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


template<class T, class IP>
void initialize(lr2<T>& lr, vector<const T*> X, vector<const T*> V, IP inner_product_X, IP inner_product_V, const Ensign::Matrix::blas_ops& blas) {

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

  orthogonalize gs(&blas);
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
template void initialize(lr2<double>& lr, vector<const double*> X, vector<const double*> V, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V, const Ensign::Matrix::blas_ops& blas);
template void initialize(lr2<double>& lr, vector<const double*> X, vector<const double*> V, double inner_product_X, double inner_product_V, const Ensign::Matrix::blas_ops& blas);
template void initialize(lr2<double>& lr, vector<const double*> X, vector<const double*> V, double* inner_product_X, double* inner_product_V, const Ensign::Matrix::blas_ops& blas);
//template void initialize(lr2<float>& lr, vector<const float*> X, vector<const float*> V, std::function<float(float*,float*)> inner_product_X, std::function<float(float*,float*)> inner_product_V, const Ensign::Matrix::blas_ops& blas);




template<class T, class IP>
void lr_add(vector<const lr2<T>*> A, const vector<T>& alpha, lr2<T>& out,
            IP inner_product_X,
            IP inner_product_V,
            const Ensign::Matrix::blas_ops& blas) {

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

  orthogonalize gs(&blas);

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


template<class T, class IP>
void lr_add(T alpha, const lr2<T>& A, T beta, const lr2<T>& B, lr2<T>& out,
            IP inner_product_X,
            IP inner_product_V,
            const Ensign::Matrix::blas_ops& blas) {

  lr_add({&A, &B}, {alpha, beta}, out, inner_product_X, inner_product_V, blas);
}

template void lr_add(double alpha, const lr2<double>& A, double beta, const lr2<double>& B, lr2<double>& out, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V, const Ensign::Matrix::blas_ops& blas);
template void lr_add(double alpha, const lr2<double>& A, double beta, const lr2<double>& B, lr2<double>& out, double inner_product_X, double inner_product_V, const Ensign::Matrix::blas_ops& blas);
template void lr_add(double alpha, const lr2<double>& A, double beta, const lr2<double>& B, lr2<double>& out, double* inner_product_X, double* inner_product_V, const Ensign::Matrix::blas_ops& blas);


template<class T, class IP>
void lr_add(T alpha, const lr2<T>& A, T beta, const lr2<T>& B,
            T gamma, const lr2<T>& C, lr2<T>& out,
            IP inner_product_X,
            IP inner_product_V,
            const Ensign::Matrix::blas_ops& blas) {

  lr_add({&A, &B, &C}, {alpha, beta, gamma}, out, inner_product_X, inner_product_V, blas);
}

  
template void lr_add(double alpha, const lr2<double>& A, double beta, const lr2<double>& B, double gamma, const lr2<double>& C, lr2<double>& out, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V, const Ensign::Matrix::blas_ops& blas);
template void lr_add(double alpha, const lr2<double>& A, double beta, const lr2<double>& B, double gamma, const lr2<double>& C, lr2<double>& out, double inner_product_X, double inner_product_V, const Ensign::Matrix::blas_ops& blas);
template void lr_add(double alpha, const lr2<double>& A, double beta, const lr2<double>& B, double gamma, const lr2<double>& C, lr2<double>& out, double* inner_product_X, double* inner_product_V, const Ensign::Matrix::blas_ops& blas);


template<class T, class IP>
void lr_mul(const lr2<T>& A, const lr2<T>& B, lr2<T>& out,
            IP inner_product_X,
            IP inner_product_V,
            const Ensign::Matrix::blas_ops& blas) {

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

  orthogonalize gs(&blas);

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

template void lr_mul(const lr2<double>& A, const lr2<double>& B, lr2<double>& out, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V, const Ensign::Matrix::blas_ops& blas);
template void lr_mul(const lr2<double>& A, const lr2<double>& B, lr2<double>& out, double inner_product_X, double inner_product_V, const Ensign::Matrix::blas_ops& blas);
template void lr_mul(const lr2<double>& A, const lr2<double>& B, lr2<double>& out, double* inner_product_X, double* inner_product_V, const Ensign::Matrix::blas_ops& blas);


template<class T>
void lr_truncate(const lr2<T>& in, lr2<T>& out, const Ensign::Matrix::blas_ops& blas) {
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

template void lr_truncate(const lr2<double>& in, lr2<double>& out, const Ensign::Matrix::blas_ops& blas);


template<class T>
double lr_inner_product(const lr2<T>& A, const lr2<T>&B, T w, const Ensign::Matrix::blas_ops& blas) {
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

template double lr_inner_product(const lr2<double>& A, const lr2<double>&B, double w, const Ensign::Matrix::blas_ops& blas);


template<class T>
double lr_norm_sq(const lr2<T>& A, const Ensign::Matrix::blas_ops& blas) {
  double norm_sq = 0.0;
  for(Index j=0;j<A.rank();j++)
    for(Index i=0;i<A.rank();i++)
      norm_sq += pow(A.S(i,j), 2);
  return norm_sq;
}

template double lr_norm_sq(const lr2<double>& A, const Ensign::Matrix::blas_ops& blas);

} // namespace Ensign