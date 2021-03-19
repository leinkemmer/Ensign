#include <generic/matrix.hpp>


template<>
void set_zero(multi_array<double,2>& a) {
    if(a.sl == stloc::host) {
        fill(a.begin(), a.end(), 0.0);
    } else {
        #ifdef __CUDACC__
        // TODO
        #else
        // ERROR!
        #endif
    }
}

template<>
void set_zero(multi_array<float,2>& a) {
    if(a.sl == stloc::host) {
        fill(a.begin(), a.end(), 0.0);
    } else {
        #ifdef __CUDACC__
        // TODO
        #else
        // ERROR!
        #endif
    }
}

//template struct multi_array<double,2>;
//template struct multi_array<float,2>;

//template void set_zero(multi_array<double,2>&);
//template void set_zero(multi_array<float,2>&);
