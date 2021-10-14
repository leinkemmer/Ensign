#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>
#include <generic/kernels.hpp>
#include <generic/matrix.hpp>

/* Class to store the low-rank factors (X, S, V) that are used in the computation.
*
* The low-rank factors can reside either on the host (CPU) or the device (GPU)
* depending on the value of stloc (either stloc::host or stloc::device).
*/
template<class T>
struct lr2 {
  multi_array<T, 2> S;
  multi_array<T, 2> X;
  multi_array<T, 2> V;

  lr2(stloc sl=stloc::host) : S(sl), X(sl), V(sl) {}

  lr2(Index r, array<Index,2> N, stloc sl=stloc::host) : S({r,r},sl), X({N[0],r},sl), V({N[1],r},sl) {}

  Index problem_size_X() const {
    return X.shape()[0];
  }
  Index problem_size_V() const {
    return V.shape()[0];
  }
  Index rank() const {
    return S.shape()[0];
  }
};

/* Initializes (an already allocated) lr2 to \sum_i X_i V_i.
*
* The rank r is taken from the initialized lr. If X.size() and V.size() are smaller
* than r, the remaining basis functions are initialized in a random manner (subject
* (to the orthogonality condition).
* X.size() == V.size() is required.
*/
template<class T>
void initialize(lr2<T>& lr, vector<const T*> X, vector<const T*> V, std::function<T(T*,T*)> inner_product_X,
                std::function<T(T*,T*)> inner_product_V, const blas_ops& blas);



/* Return an inner product function object for use in, e.g., in gram_schmidt.
*/
template<class T>
std::function<T(T*,T*)> inner_product_from_const_weight(T w, Index N);

/* Return an inner product function object for use in, e.g., in gram_schmidt.
*
*/
template<class T>
std::function<T(T*,T*)> inner_product_from_weight(const T* w, Index N);


/* Orthogonalizes Q and returns the corresponding upper triangular matrix R.
*
* Note that the inputs are overwritten.
*/
struct gram_schmidt {

  gram_schmidt(const blas_ops* _blas);
  ~gram_schmidt();

  void operator()(multi_array<double,2>& Q, multi_array<double,2>& R, std::function<double(double*,double*)> inner_product);
  void operator()(multi_array<double,2>& Q, multi_array<double,2>& R, double w);
  
private:
  const blas_ops* blas;
  #ifdef __CUDACC__
  curandGenerator_t gen;
  #endif
};
