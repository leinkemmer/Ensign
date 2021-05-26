#include <generic/kernels.hpp>
#include <lr/lr.hpp>
#include <generic/matrix.hpp>

#ifdef __CUDACC__
__global__ void dmaxpy(int n, double* a, double* x, double* y){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    y[idx] = -(*a)*x[idx] + y[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void scale_unique(double* x, double alpha){
  *x *= alpha;
}

__global__ void scale_sqrt_unique(double* x, double alpha){
  *x = sqrt(*x * alpha);
}

__global__ void ptw_div_gs(int n, double* A, double* alpha){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    if(abs(*alpha) > 1e-12){
      A[idx] /= (*alpha);
    }
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void ptw_mult(int n, double* A, double* B, double* C){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    C[idx] = A[idx] * B[idx];
    idx += blockDim.x * gridDim.x;
  }
}
#endif


template<class T>
std::function<T(T*,T*)> inner_product_from_weight(T* w, Index N) {
  return [w,N](T* a, T*b) {
    T result=T(0.0);
    for(Index i=0;i<N;i++)
    result += w[i]*a[i]*b[i];
    return result;
  };
};
template std::function<double(double*,double*)> inner_product_from_weight(double* w, Index N);
template std::function<float(float*,float*)> inner_product_from_weight(float* w, Index N);


template<class T>
std::function<T(T*,T*)> inner_product_from_const_weight(T w, Index N) {
  return [w,N](T* a, T*b) {
    T result=T(0.0);
    for(Index i=0;i<N;i++){
      result += a[i]*b[i];
    }
    result *= w;
    return result;
  };
};
template std::function<double(double*,double*)> inner_product_from_const_weight(double w, Index N);
template std::function<float(float*,float*)> inner_product_from_const_weight(float w, Index N);

template<>
void gram_schmidt(multi_array<double,2>& Q, multi_array<double,2>& R, std::function<double(double*,double*)> inner_product) {
  array<Index,2> dims = Q.shape();
  for(Index j=0;j<dims[1];j++) {
    for(Index k=0;k<j;k++) {
      R(k,j) = inner_product(Q.extract({j}), Q.extract({k}));
      cblas_daxpy(dims[0], -R(k,j), Q.extract({k}), 1, Q.extract({j}),1);
      R(j,k) = 0.0;
    }
    R(j,j) = sqrt(inner_product(Q.extract({j}), Q.extract({j})));

    if(std::abs(R(j,j)) < 1e-12){
    //  cout << "Warning: linearly dependent columns in Gram-Schmidt" << endl;
    } else{
      cblas_dscal(dims[0],1.0/R(j,j),Q.extract({j}),1);
    }
  }
};

#ifdef __CUDACC__
void gram_schmidt_gpu(multi_array<double,2>& Q, multi_array<double,2>& R, double w) { //with constant weight for inner product
  cublasHandle_t  handle;
  cublasCreate (&handle);

  Index n = Q.shape()[0];
  int r = Q.shape()[1];

  for(Index j=0;j<r;j++) {
    for(Index k=0;k<j;k++) {
      cublasDdot (handle, n, &Q(0,j), 1, &Q(0,k), 1,&R(k,j));
      scale_unique<<<1,1>>>(&R(k,j),w); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
      dmaxpy<<<2,2>>>(n, &R(k,j), &Q(0,k), &Q(0,j));
      scale_unique<<<1,1>>>(&R(j,k),0.0);
    }
      cublasDdot (handle, n, &Q(0,j), 1, &Q(0,j), 1, &R(j,j));
      scale_sqrt_unique<<<1,1>>>(&R(j,j),w);
      ptw_div_gs<<<2,2>>>(n, &Q(0,j), &R(j,j));
  }
  cublasDestroy(handle);
};

// STILL TO BE TESTED
void gram_schmidt_gpu(multi_array<double,2>& Q, multi_array<double,2>& R, double* w) { //with non-constant weight for inner product. Still to be tested
  cublasHandle_t  handle;
  cublasCreate (&handle);

  Index n = Q.shape()[0];
  int r = Q.shape()[1];
  multi_array<double,1> tmp({n},stloc::device);

  for(Index j=0;j<r;j++) {
    for(Index k=0;k<j;k++) {
      ptw_mult<<<2,2>>>(n, &Q(0,j),w,tmp.begin());
      cublasDdot (handle, n, tmp.begin(), 1, &Q(0,k), 1,&R(k,j));
      dmaxpy<<<2,2>>>(n, &R(k,j), &Q(0,k), &Q(0,j));
      scale_unique<<<1,1>>>(&R(j,k),0.0);
    }
      ptw_mult<<<2,2>>>(n, &Q(0,j),w,tmp.begin());
      cublasDdot (handle, n, &Q(0,j), 1, tmp.begin(), 1,&R(j,j));
      ptw_div_gs<<<2,2>>>(n, &Q(0,j), &R(j,j));

  }
  cublasDestroy(handle);

};

#endif
/*
void gram_schmidt_gpu(Index n, int r, double* Q, double* R, double w) { //with constant weight for inner product
  cublasHandle_t  handle;
  cublasCreate (&handle);

  for(Index j=0;j<r;j++) {
    for(Index k=0;k<j;k++) {
      cublasDdot (handle, n, Q + n*j, 1, Q + n*k, 1,R + k + r*j);
      scale_unique<<<1,1>>>(R + k + r*j,w); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
      dmaxpy<<<2,2>>>(n, R + k + r*j, Q + n*k, Q + n*j);
      scale_unique<<<1,1>>>(R + j + r*k,0.0);
    }
      cublasDdot (handle, n, Q + n*j, 1, Q + n*j, 1,R + j + r*j);
      scale_sqrt_unique<<<1,1>>>(R + j + r*j,w);
      ptw_div_gs<<<2,2>>>(n, Q + n*j, R + j + r*j);

  }

  cublasDestroy(handle);

};
*/

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
};

/*
template<class T>
void gram_schmidt(multi_array<T,2>& Q, multi_array<T,2>& R,
std::function<T(T*,T*)> inner_product) {

array<Index,2> dims = Q.shape();

for(Index j=0;j<dims[1];j++) {
for(int k=0;k<j;k++) {
R(k,j) = inner_product(Q.extract(j), Q.extract(k));
Q.extract(k) *= R(k,j);
Q.extract(j) -= Q.extract(k);
}
R(j,j) = sqrt(inner_product(Q.extract(j), Q.extract(j)));
Q.extract(j) /= R(j,j);
}

};
*/
/*
template<class T>
void ptw_prod_diff(T* v1, Index n, T* v2, T scal) {
  std::transform(v1, v1+n,v2, v1, [&scal](T& a,T& b){return a-scal*b;} );
};
template void ptw_prod_diff(double* v1, Index n, double* v2, double scal);
template void ptw_prod_diff(float* v1, Index n, float* v2, float scal);

template<class T>
void gram_schmidt(multi_array<T,2>& Q, multi_array<T,2>& R, std::function<T(T*,T*)> inner_product) {

  array<Index,2> dims = Q.shape();

  for(Index j=0;j<dims[1];j++) {
    for(Index k=0;k<j;k++) {
      R(k,j) = inner_product(Q.extract({j}), Q.extract({k}));
      ptw_prod_diff(Q.extract({j}), dims[0], Q.extract({k}), R(k,j));
    }
    R(j,j) = sqrt(inner_product(Q.extract({j}), Q.extract({j})));
    if(std::abs(R(j,j)) < T(10)*std::numeric_limits<T>::epsilon()){
      cout << "Warning: linearly dependent columns in Gram-Schmidt" << endl;
    } else{
      ptw_div(Q.extract({j}), dims[0], R(j,j));
    }
  }

};
template void gram_schmidt(multi_array<double,2>& Q, multi_array<double,2>& R, std::function<double(double*,double*)> inner_product);
template void gram_schmidt(multi_array<float,2>& Q, multi_array<float,2>& R, std::function<float(float*,float*)> inner_product);
*/
/*
template<class T>
void ptw_div(T* v, Index n, T scal) {
  std::transform(v, v+n, v, [&scal](T& a){return a/scal;} );
};
template void ptw_div(double* v, Index n, double scal);
template void ptw_div(float* v, Index n, float scal);
*/
template<class T>
void initialize(lr2<T>& lr, vector<T*> X, vector<T*> V, std::function<T(T*,T*)> inner_product_X, std::function<T(T*,T*)> inner_product_V) {

  int n_b = X.size();
  Index r = lr.rank();

  std::default_random_engine generator(time(0));
  std::normal_distribution<double> distribution(0.0,1.0);

  for(Index k=0;k<r;k++) {
    if(k < n_b){
      for(Index i=0;i<lr.problem_size_X();i++) {
        lr.X(i, k) = X[k][i];
      }
      for(Index i=0;i<lr.problem_size_V();i++) {
        lr.V(i, k) = V[k][i];
      }
    }
    else{
      for(Index i=0;i<lr.problem_size_X();i++) {
        lr.X(i, k) = distribution(generator);
        //lr.X(i,k) = i;
      }
      for(Index i=0;i<lr.problem_size_V();i++) {
        lr.V(i, k) = distribution(generator);
        //lr.V(i,k) = i;
      }
    }
  }

  multi_array<T, 2> X_R(lr.S.shape()), V_R(lr.S.shape());

  gram_schmidt(lr.X, X_R, inner_product_X);
  gram_schmidt(lr.V, V_R, inner_product_V);

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

  matmul_transb(X_R, V_R, lr.S);

};
template void initialize(lr2<double>& lr, vector<double*> X, vector<double*> V, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V);
template void initialize(lr2<float>& lr, vector<float*> X, vector<float*> V, std::function<float(float*,float*)> inner_product_X, std::function<float(float*,float*)> inner_product_V);
