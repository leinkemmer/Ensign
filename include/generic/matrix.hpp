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

/* Additions from kinetic-cme
*/
namespace Tensor {

template <size_t m, size_t d, class T>
void matricize(const multi_array<T, d>& input, multi_array<T, 2>& output)
{
    std::array<Index, d> shape{input.shape()};
    std::array<Index, d - 1> cols_shape, vec_index_cols;
    remove_element(std::begin(shape), std::end(shape), std::begin(cols_shape), m);
    std::vector<Index> vec_index(d, 0);
    Index i, j;

    assert(shape[m] == output.shape()[1] && prod(cols_shape) == output.shape()[0]);

    for (auto const& el : input) {
        i = vec_index[m];
        remove_element(std::begin(vec_index), std::end(vec_index),
                      std::begin(vec_index_cols), m);
        j = IndexFunction::VecIndexToCombIndex(std::begin(vec_index_cols),
                                               std::end(vec_index_cols),
                                               std::begin(cols_shape));
        output(j, i) = el;
        IndexFunction::IncrVecIndex(std::begin(shape), std::begin(vec_index),
                                    std::end(vec_index));
    }
}

template <>
void matricize<0, 3, double>(const multi_array<double, 3>& input,
                             multi_array<double, 2>& output);
template <>
void matricize<1, 3, double>(const multi_array<double, 3>& input,
                             multi_array<double, 2>& output);
template <>
void matricize<2, 3, double>(const multi_array<double, 3>& input,
                             multi_array<double, 2>& output);

template <size_t m, size_t d, class T>
void tensorize(const multi_array<T, 2>& input, multi_array<T, d>& output)
{
    std::array<Index, d> shape{output.shape()};
    std::array<Index, d - 1> cols_shape, vec_index_cols;
    remove_element(std::begin(shape), std::end(shape), std::begin(cols_shape), m);
    std::vector<Index> vec_index(d, 0);
    Index i, j;

    assert(shape[m] == input.shape()[1] && prod(cols_shape) == input.shape()[0]);

    for (auto& el : output) {
        i = vec_index[m];
        remove_element(std::begin(vec_index), std::end(vec_index),
                      std::begin(vec_index_cols), m);
        j = IndexFunction::VecIndexToCombIndex(std::begin(vec_index_cols),
                                               std::end(vec_index_cols),
                                               std::begin(cols_shape));
        el = input(j, i);
        IndexFunction::IncrVecIndex(std::begin(shape), std::begin(vec_index),
                                    std::end(vec_index));
    }
}

template <>
void tensorize<0, 3, double>(const multi_array<double, 2>& input,
                             multi_array<double, 3>& output);
template <>
void tensorize<1, 3, double>(const multi_array<double, 2>& input,
                             multi_array<double, 3>& output);
template <>
void tensorize<2, 3, double>(const multi_array<double, 2>& input,
                             multi_array<double, 3>& output);
} // namespace Tensor

} // namespace Ensign