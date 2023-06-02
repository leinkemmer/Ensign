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

  void resize(Index r, array<Index,2> N) {
    S.resize({r,r});
    X.resize({N[0],r});
    V.resize({N[1],r});
  }

  Index size_X() const {
    return X.shape()[0];
  }
  Index size_V() const {
    return V.shape()[0];
  }
  Index rank() const {
    return S.shape()[0];
  }

  /* Computes the full matrix (can be very expensive and should not be used in production code)
  */
  multi_array<T,2> full(const blas_ops& blas) const {
    multi_array<T, 2> K = X;
    blas.matmul(X, S, K);
    multi_array<T, 2> out({size_X(), size_V()});
    blas.matmul_transb(K, V, out);
    return out;
  }
};


/*  Add two low-rank representations together.
*
*   Note that no truncation is performed by this function. To do that call lr_truncate.
*/
template<class T>
void lr_add(T alpha, const lr2<T>& A, T beta, const lr2<T>& B, lr2<T>& out,
            std::function<T(T*,T*)> inner_product_X,
            std::function<T(T*,T*)> inner_product_V,
            const blas_ops& blas);

/*  Add three low-rank representations together.
*
*   Note that no truncation is performed by this function. To do that call lr_truncate.
*/
template<class T>
void lr_add(T alpha, const lr2<T>& A, T beta, const lr2<T>& B,
            T gamma, const lr2<T>& C, lr2<T>& out,
            std::function<T(T*,T*)> inner_product_X,
            std::function<T(T*,T*)> inner_product_V,
            const blas_ops& blas);


/* Multiplies two low-rank representations together.
*
*  Note that no truncation is performed by this function. To do that call lr_truncate.
*/
template<class T>
void lr_mul(const lr2<T>& A, const lr2<T>& B, lr2<T>& out,
            std::function<T(T*,T*)> inner_product_X,
            std::function<T(T*,T*)> inner_product_V,
            const blas_ops& blas);

/*  Truncate a low-rank representation inplace to a fixed rank r.
*
*   The rank is determined by the size of out.
*/
template<class T>
void lr_truncate(const lr2<T>& in, lr2<T>& out, const blas_ops& blas);


/* Computes the inner product of two low-rank representations.
*
*  Computes the inner product of two low-rank representations.
*/
template<class T>
double lr_inner_product(const lr2<T>& A, const lr2<T>&B, T w, const blas_ops& blas);


/* Computes the norm of a low-rank representations.
*
*  This assumes that X and V are orthogonalized. Thus, no inner product needs
*  to be passed to this function. The function simply computes
*  \sum_{ij} S_{ij}^2
*/
template<class T>
double lr_norm_sq(const lr2<T>& A, const blas_ops& blas);




/* Initializes (an already allocated) lr2 to \sum_i X_i V_i.
*
* The rank r is taken from the initialized lr. If X.size() and V.size() are smaller
* than r, the remaining basis functions are initialized in a random manner (subject
* (to the orthogonality condition).
* X.size() == V.size() is required.
*/
template<class T>
void initialize(lr2<T>& lr, vector<const T*> X, vector<const T*> V,
                std::function<T(T*,T*)> inner_product_X,
                std::function<T(T*,T*)> inner_product_V,
                const blas_ops& blas);


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
