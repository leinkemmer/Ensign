#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>
#include <generic/kernels.hpp>
#include <generic/index.hpp>
#include <generic/utility.hpp>

namespace Ensign{

/* Set matrix to zero
*/
template<class T>
void set_zero(multi_array<T,2>& a);

/* Set matrix to identity
*/
template<class T>
void set_identity(multi_array<T,2>& a);

/* Set vector or matrix to a constant value
*/
template<class T, size_t d>
void set_const(multi_array<T,d>& a, T alpha);


/* Multiply every row of a matrix with a vector (i.e. diag(w)a)  
*/
template<class T>
void ptw_mult_row(const multi_array<T,2>& a, const multi_array<T,1>& w, multi_array<T,2>& out);

/* Transpose a square matrix inplace (intended for small matrices)
*/
template<class T>
void transpose_inplace(multi_array<T,2>& a);


struct blas_ops {

  blas_ops(bool gpu = false);
  ~blas_ops();

  /* Matrix multiplication (computes c = a b)
  *
  * The number of rows in matrix c can be smaller than the number of rows in matrix a. In this
  * case only a partial result is computed.
  */
  template<class T>
  void matmul(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c) const;

  /* Matrix multiplication with first matrix transposed (computes c = a^T b)
  */
  template<class T>
  void matmul_transa(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c) const;

  /* Matrix multiplication with second matrix transposed (computes c = a b^T)
  */
  template<class T>
  void matmul_transb(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c) const;

  /* Matrix multiplication with both matrices transposed (computes c = a^T b^T)
  */
  template<class T>
  void matmul_transab(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c) const;

  /* Matrix vector multiplication (c = a b)
  */
  template<class T>
  void matvec(const multi_array<T,2>& a, const multi_array<T,1>& b, multi_array<T,1>& c) const;

  /* Matrix vector multiplication with matrix transposed (c = a^T b)
  */
  template<class T>
  void matvec_trans(const multi_array<T,2>& a, const multi_array<T,1>& b, multi_array<T,1>& c) const;

  bool gpu;
  #ifdef __CUDACC__
  cublasHandle_t  handle;
  cublasHandle_t  handle_devres; // cuBLAS routines return scalar results on device
  cusolverDnHandle_t handle_cusolver;
  #endif
};

/* Helper class to perform diagonalization of a symmetric matrix (also known as the Schur decomposition)
   Factorizes the matrix CC as follows
   CC = TT*diag(diag_r)*TT^T
*/
struct diagonalization {
  diagonalization(Index m);

  void operator()(const multi_array<double,2>& CC, multi_array<double,2>& TT, multi_array<double,1>& diag_r);

private:
  #ifdef __MKL__
  MKL_INT lwork;
  #else
  int lwork; 
  #endif
};

/* Computes the singular value decomposition of the matrix input
*/
template<class T>
void svd(const multi_array<T,2>& input, multi_array<T,2>& U, multi_array<T,2>& V, multi_array<T,1>& sigma_diag, const blas_ops& blas);

// namespace Matrix

} // namespace Ensign