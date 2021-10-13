#include <generic/kernels.hpp>
#include <lr/lr.hpp>
#include <generic/matrix.hpp>

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


template<class T>
std::function<T(T*,T*)> inner_product_from_const_weight(T w, Index N) {
  return [w,N](T* a, T*b) {
    T result=T(0.0);
    #ifdef __OPENMP__
    #pragma omp parallel for reduction(+:result)
    #endif
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

  std::default_random_engine generator(time(0));
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
    } else{

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
void gram_schmidt_gpu(multi_array<double,2>& Q, multi_array<double,2>& R, double w, curandGenerator_t gen) { //with constant weight for inner product
  Index n = Q.shape()[0];
  int r = Q.shape()[1];
  double* nrm;
  cudaMalloc((void**)&nrm,sizeof(double));

  for(Index j=0;j<r;j++) {
    for(Index k=0;k<j;k++) {
      cublasDdot (handle_dot, n, &Q(0,j), 1, &Q(0,k), 1,&R(k,j));
      cudaDeviceSynchronize();
      scale_unique<<<1,1>>>(&R(k,j),w); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
      dmaxpy<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &R(k,j), &Q(0,k), &Q(0,j));
      scale_unique<<<1,1>>>(&R(j,k),0.0);
    }

      cublasDdot (handle_dot, n, &Q(0,j), 1, &Q(0,j), 1, &R(j,j));
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
          cublasDdot (handle_dot, n, &Q(0,j), 1, &Q(0,k), 1, nrm);
          cudaDeviceSynchronize();
          scale_unique<<<1,1>>>(nrm,w); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
          dmaxpy<<<(n+n_threads-1)/n_threads,n_threads>>>(n, nrm, &Q(0,k), &Q(0,j));
        }

          cublasDdot (handle_dot, n, &Q(0,j), 1, &Q(0,j), 1, nrm);
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
      cublasDdot (handle_dot, n, tmp.begin(), 1, &Q(0,k), 1,&R(k,j));
      cudaDeviceSynchronize();
      dmaxpy<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &R(k,j), &Q(0,k), &Q(0,j));
      scale_unique<<<1,1>>>(&R(j,k),0.0);
    }
      ptw_mult<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &Q(0,j),w,tmp.begin());
      cublasDdot (handle_dot, n, &Q(0,j), 1, tmp.begin(), 1,&R(j,j));
      cudaDeviceSynchronize();
      ptw_div_gs<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &Q(0,j), &R(j,j));

  }
};
*/

#endif

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
void initialize(lr2<T>& lr, vector<const T*> X, vector<const T*> V, std::function<T(T*,T*)> inner_product_X, std::function<T(T*,T*)> inner_product_V) {

  int n_b = X.size();
  Index r = lr.rank();

  std::default_random_engine generator(time(0));
  std::normal_distribution<double> distribution(0.0,1.0);

  for(Index k=0;k<r;k++) {
    if(k < n_b){
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0;i<lr.problem_size_X();i++) {
        lr.X(i, k) = X[k][i];
      }
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0;i<lr.problem_size_V();i++) {
        lr.V(i, k) = V[k][i];
      }
    }
    else{
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0;i<lr.problem_size_X();i++) {
        lr.X(i, k) = distribution(generator);
      }
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0;i<lr.problem_size_V();i++) {
        lr.V(i, k) = distribution(generator);
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
template void initialize(lr2<double>& lr, vector<const double*> X, vector<const double*> V, std::function<double(double*,double*)> inner_product_X, std::function<double(double*,double*)> inner_product_V);
//template void initialize(lr2<float>& lr, vector<const float*> X, vector<const float*> V, std::function<float(float*,float*)> inner_product_X, std::function<float(float*,float*)> inner_product_V);
