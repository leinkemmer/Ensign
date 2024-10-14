#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>

namespace Ensign{

template<size_t d, size_t dim>
struct fft {

    fft(array<Index,dim> dims_, multi_array<double,d>& real, multi_array<complex<double>,d>& freq);
    ~fft();

    void forward(multi_array<double,d>& real, multi_array<complex<double>,d>& freq); 
    void backward(multi_array<complex<double>,d>& freq, multi_array<double,d>& real);

private:
    array<fftw_plan,2> plans;
    #ifdef __CUDACC__
    array<cufftHandle,2> cuda_plans;
    #endif

    void set_null() {
      plans[0] = 0;
      plans[1] = 0;
      #ifdef __CUDACC__
      cuda_plans[0] = 0;
      cuda_plans[1] = 0;
      #endif
    }
};

template<size_t d>
using fft3d = fft<d, 3>;

template<size_t d>
using fft2d = fft<d, 2>;

template<size_t d>
using fft1d = fft<d, 1>;


/* Helper routines to create FFTW plans for 1d transforms
*/
array<fftw_plan,2> create_plans_1d(Index dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq);
array<fftw_plan,2> create_plans_1d(Index dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq);

/* Helper routines to create FFTW plans for 2d transforms
*/
array<fftw_plan,2> create_plans_2d(array<Index,2> dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq);
array<fftw_plan,2> create_plans_2d(array<Index,2> dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq);

/* Helper routines to create FFTW plans for 3d transforms
*/
array<fftw_plan,2> create_plans_3d(array<Index,3> dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq);
array<fftw_plan,2> create_plans_3d(array<Index,3> dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq);

/* Helper function to destroy FFTW plans
*/
void destroy_plans(array<fftw_plan,2>& plans);


#ifdef __CUDACC__
/* Helper routines to create cuFFT plans for 1d transforms.
*/
array<cufftHandle,2> create_plans_1d(Index dims_, int howmany);

/* Helper routines to create cuFFT plans for 2d transforms.
*/
array<cufftHandle,2> create_plans_2d(array<Index,2> dims_, int howmany);

/* Helper routines to create cuFFT plans for 3d transforms.
*/
array<cufftHandle,2> create_plans_3d(array<Index,3> dims_, int howmany);

/* Helper function to destroy cuFFT plans
*/
void destroy_plans(array<cufftHandle,2>& plans);
#endif

} // namespace Ensign