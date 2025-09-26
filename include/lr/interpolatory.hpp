#pragma once

#include <generic/common.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/lr.hpp>

namespace Ensign {

void deim_ext(const multi_array<double,2>& _U, Index r, multi_array<Index,1>& I);

// returns the condition number
double colloquation_to_lr(const multi_array<double,2>& f_I, multi_array<double,2>& f_J, const multi_array<Index,1>& I, multi_array<double,2>& X, multi_array<double,2>& L, Matrix::blas_ops& blas);

// returns the condition number
double colloquation_to_lr(const multi_array<double,2>& f_I, multi_array<double,2>& f_J, const multi_array<Index,1>& I, lr2<double>& f, Matrix::blas_ops& blas);

}