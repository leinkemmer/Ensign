#include <generic/tensor.hpp>

namespace Ensign {

namespace Tensor {

/* Additions from kinetic-cme
*/
template <>
void matricize<0, 3, double>(const multi_array<double, 3>& input,
                                     multi_array<double, 2>& output)
{
    const auto n = input.shape();
    for (Index k = 0; k < n[2]; ++k) {
        for (Index j = 0; j < n[1]; ++j) {
            for (Index i = 0; i < n[0]; ++i) {
                output(j + n[1] * k, i) = input(i, j, k);
            }
        }
    }
}

template <>
void matricize<1, 3, double>(const multi_array<double, 3>& input,
                                     multi_array<double, 2>& output)
{
    const auto n = input.shape();
    for (Index k = 0; k < n[2]; ++k) {
        for (Index j = 0; j < n[1]; ++j) {
            for (Index i = 0; i < n[0]; ++i) {
                output(k + n[2] * i, j) = input(i, j, k);
            }
        }
    }
}

template <>
void matricize<2, 3, double>(const multi_array<double, 3>& input,
                                     multi_array<double, 2>& output)
{
    const auto n = input.shape();
    for (Index k = 0; k < n[2]; ++k) {
        for (Index j = 0; j < n[1]; ++j) {
            for (Index i = 0; i < n[0]; ++i) {
                output(i + n[0] * j, k) = input(i, j, k);
            }
        }
    }
}

template <>
void tensorize<0, 3, double>(const multi_array<double, 2>& input,
                                     multi_array<double, 3>& output)
{
    const auto n = output.shape();
    for (Index k = 0; k < n[2]; ++k) {
        for (Index j = 0; j < n[1]; ++j) {
            for (Index i = 0; i < n[0]; ++i) {
                output(i, j, k) = input(j + n[1] * k, i);
            }
        }
    }
}

template <>
void tensorize<1, 3, double>(const multi_array<double, 2>& input,
                                     multi_array<double, 3>& output)
{
    const auto n = output.shape();
    for (Index k = 0; k < n[2]; ++k) {
        for (Index j = 0; j < n[1]; ++j) {
            for (Index i = 0; i < n[0]; ++i) {
                output(i, j, k) = input(k + n[2] * i, j);
            }
        }
    }
}

template <>
void tensorize<2, 3, double>(const multi_array<double, 2>& input,
                                     multi_array<double, 3>& output)
{
    const auto n = output.shape();
    for (Index k = 0; k < n[2]; ++k) {
        for (Index j = 0; j < n[1]; ++j) {
            for (Index i = 0; i < n[0]; ++i) {
                output(i, j, k) = input(i + n[0] * j, k);
            }
        }
    }
}

} // namespace Tensor

} // namespace Ensign