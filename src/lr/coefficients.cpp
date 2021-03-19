#include <lr/coefficients.hpp>


template<size_t d1, size_t d2>
void coeff_double(multi_array<double,d1>& a, multi_array<double,d2>& b, double w, multi_array<double,d1+d2-2>& out) {

}


template<>
void coeff(multi_array<double,2>& a, multi_array<double,2>& b, double w, multi_array<double,2>& out) {
    coeff_double<2,2>(a, b, w, out);
}


