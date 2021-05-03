#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <cblas.h>
#include <random>


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
    if(std::abs(R(j,j)) < double(1000)*std::numeric_limits<double>::epsilon()){
      cout << "Warning: linearly dependent columns in Gram-Schmidt" << endl;
    } else{
      cblas_dscal(dims[0],1.0/R(j,j),Q.extract({j}),1);
    }
  }
};

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
