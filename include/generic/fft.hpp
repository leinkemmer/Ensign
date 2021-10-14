#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>

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
