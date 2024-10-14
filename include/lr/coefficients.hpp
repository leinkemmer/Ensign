#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>
#include <generic/matrix.hpp>

namespace Ensign{

/* Computes a low-rank coefficient that approximates \int a_i(x) b_j(x) \,dx.
 *
 * This functions assumes that w is a constant weight.
*/
template<class T>
void coeff(const multi_array<T,2>& a, const multi_array<T,2>& b, T w, multi_array<T,2>& out, const blas_ops& blas);

/* Computes a low-rank coefficient that approximates \int a_i(x) b_j(x) \,dx.
 *
 * If a and b are of size (n,r) then w must be of size n.
 * The weight w can be used as a pure weight (in a quadrature sense) but also
 * can incooperate a x dependent function.
*/
template<class T>
void coeff(const multi_array<T,2>& a, const multi_array<T,2>& b, const multi_array<T,1>& w, multi_array<T,2>& out, const blas_ops& blas);

/* Computes an approximation to the integral \int a_i(x) \,dx.
*
* This function assumes that w is a constant weight.
*/
template<class T>
void integrate(const multi_array<T,2>& a, T w, multi_array<T,1>& out, const blas_ops& blas);

/* Computes an approximation to the integral \int a_i(x) \,dx.
*
*  If a is of size (n,r) then w must be a contiguous memory region of size n.
*  The weight w can be used as a pure weight (in a quadrature sense) but also
*  can incooperate a x dependent function.
*/
template<class T>
void integrate(const multi_array<T,2>& a, const multi_array<T,1>& w, multi_array<T,1>& out, const blas_ops& blas);


/* Computes a low-rank coefficient that approximates \int a_i(x) b_j(x) b_k(x) \,dx.
 *
 * This function assumes that w is a constant weight.
*/
template<class T>
void coeff(const multi_array<T,2>& a, const multi_array<T,2>& b, const multi_array<T,2>& c, T w, multi_array<T,3>& out, const blas_ops& blas);

} // namespace Ensign