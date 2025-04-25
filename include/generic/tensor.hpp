#pragma once

#include <lr/lr.hpp>

namespace Ensign {

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
        j = IndexFunction::vec_index_to_comb_index(std::begin(vec_index_cols),
                                               std::end(vec_index_cols),
                                               std::begin(cols_shape));
        output(j, i) = el;
        IndexFunction::incr_vec_index(std::begin(shape), std::begin(vec_index),
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
        j = IndexFunction::vec_index_to_comb_index(std::begin(vec_index_cols),
                                               std::end(vec_index_cols),
                                               std::begin(cols_shape));
        el = input(j, i);
        IndexFunction::incr_vec_index(std::begin(shape), std::begin(vec_index),
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

template <class T>
multi_array<T, 2> orthogonalize(multi_array<T, 2>& input,
                                        const Index n_basisfunctions, const T weight,
                                        const Ensign::Matrix::blas_ops& blas)
{
    Index rows = input.shape()[0];
    Index cols = input.shape()[1];

    assert(n_basisfunctions <= cols);

    std::default_random_engine generator(1234);
    std::normal_distribution<double> distribution(0.0, 1.0);

#ifdef __OPENMP__
#pragma omp parallel for
#endif
    for (Index k = n_basisfunctions; k < cols; ++k) {
        for (Index i = 0; i < rows; ++i) {
            input(i, k) = distribution(generator);
        }
    }

    multi_array<T, 2> R({cols, cols});

    Ensign::orthogonalize gs(&blas);
    gs(input, R, weight);

    for (Index j = n_basisfunctions; j < cols; ++j) {
        for (Index i = 0; i < cols; ++i) {
            R(i, j) = T(0.0);
        }
    }
    return R;
}

} // namespace Tensor

}