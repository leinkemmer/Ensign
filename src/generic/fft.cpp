#include <generic/fft.hpp>
#ifdef __CUDACC__
#include <cufft.h>
#endif

array<fftw_plan,2> create_plans_1d(Index dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq){
  array<fftw_plan,2> out;
  int dims = int(dims_);

  out[0] = fftw_plan_many_dft_r2c(1, &dims, real.shape()[1], real.begin(), NULL, 1, dims, (fftw_complex*)freq.begin(), NULL, 1, dims/2 + 1, FFTW_ESTIMATE);
  out[1] = fftw_plan_many_dft_c2r(1, &dims, real.shape()[1], (fftw_complex*)freq.begin(), NULL, 1, dims/2 + 1, real.begin(), NULL, 1, dims, FFTW_ESTIMATE);

  return out;
}

array<fftw_plan,2> create_plans_1d(Index dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq){
  array<fftw_plan,2> out;
  int dims = int(dims_);

  out[0] = fftw_plan_many_dft_r2c(1, &dims, 1, real.begin(), NULL, 1, dims, (fftw_complex*)freq.begin(), NULL, 1, dims/2 + 1, FFTW_ESTIMATE);
  out[1] = fftw_plan_many_dft_c2r(1, &dims, 1, (fftw_complex*)freq.begin(), NULL, 1, dims/2 + 1, real.begin(), NULL, 1, dims, FFTW_ESTIMATE);

  return out;
}

array<fftw_plan,2> create_plans_2d(array<Index,2> dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq){
  array<fftw_plan,2> out;
  array<int,2> dims = {int(dims_[1]),int(dims_[0])};

  out[0] = fftw_plan_many_dft_r2c(2, dims.begin(), real.shape()[1], real.begin(), NULL, 1, dims[1]*dims[0], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), FFTW_ESTIMATE);
  out[1] = fftw_plan_many_dft_c2r(2, dims.begin(), real.shape()[1], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), real.begin(), NULL, 1, dims[1]*dims[0], FFTW_ESTIMATE);

  return out;
}

array<fftw_plan,2> create_plans_2d(array<Index,2> dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq){
  array<fftw_plan,2> out;
  array<int,2> dims = {int(dims_[1]),int(dims_[0])};

  out[0] = fftw_plan_many_dft_r2c(2, dims.begin(), 1, real.begin(), NULL, 1, dims[1]*dims[0], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), FFTW_ESTIMATE);
  out[1] = fftw_plan_many_dft_c2r(2, dims.begin(), 1, (fftw_complex*)freq.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), real.begin(), NULL, 1, dims[1]*dims[0], FFTW_ESTIMATE);

  return out;
}

array<fftw_plan,2> create_plans_3d(array<Index,3> dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq){
  array<fftw_plan,2> out;
  array<int,3> dims = {int(dims_[2]), int(dims_[1]), int(dims_[0])};

  out[0] = fftw_plan_many_dft_r2c(3, dims.begin(), real.shape()[1], real.begin(), NULL, 1, dims[2]*dims[1]*dims[0], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), FFTW_ESTIMATE);
  out[1] = fftw_plan_many_dft_c2r(3, dims.begin(), real.shape()[1], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), real.begin(), NULL, 1, dims[2]*dims[1]*dims[0], FFTW_ESTIMATE);

  return out;
}

array<fftw_plan,2> create_plans_3d(array<Index,3> dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq){
  array<fftw_plan,2> out;
  array<int,3> dims = {int(dims_[2]), int(dims_[1]), int(dims_[0])};

  out[0] = fftw_plan_many_dft_r2c(3, dims.begin(), 1, real.begin(), NULL, 1, dims[2]*dims[1]*dims[0], (fftw_complex*)freq.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), FFTW_ESTIMATE);
  out[1] = fftw_plan_many_dft_c2r(3, dims.begin(), 1, (fftw_complex*)freq.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), real.begin(), NULL, 1, dims[2]*dims[1]*dims[0], FFTW_ESTIMATE);

  return out;
}

void destroy_plans(array<fftw_plan,2>& plans){
  fftw_destroy_plan(plans[0]);
  fftw_destroy_plan(plans[1]);
}



#ifdef __CUDACC__
array<cufftHandle,2> create_plans_1d(Index dims_, int howmany){
  array<cufftHandle,2> out;
  int dims = int(dims_);

  cufftPlanMany(&out[0], 1, &dims, NULL, 1, dims, NULL, 1, dims/2 + 1, CUFFT_D2Z, howmany);
  cufftPlanMany(&out[1], 1, &dims, NULL, 1, dims/2 + 1, NULL, 1, dims, CUFFT_Z2D, howmany);

  return out;
}

array<cufftHandle,2> create_plans_2d(array<Index,2> dims_, int howmany){
  array<cufftHandle,2> out;
  array<int,2> dims = {int(dims_[1]),int(dims_[0])};

  cufftPlanMany(&out[0], 2, dims.begin(), NULL, 1, dims[1]*dims[0], NULL, 1, dims[0]*(dims[1]/2 + 1), CUFFT_D2Z, howmany);
  cufftPlanMany(&out[1], 2, dims.begin(), NULL, 1, dims[0]*(dims[1]/2 + 1), NULL, 1, dims[1]*dims[0], CUFFT_Z2D, howmany);

  return out;
}

array<cufftHandle,2> create_plans_3d(array<Index,3> dims_, int howmany){
  array<cufftHandle,2> out;
  array<int,3> dims = {int(dims_[2]),int(dims_[1]),int(dims_[0])};

  cufftPlanMany(&out[0], 3, dims.begin(), NULL, 1, dims[2]*dims[1]*dims[0], NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), CUFFT_D2Z, howmany);
  cufftPlanMany(&out[1], 3, dims.begin(), NULL, 1, dims[0]*dims[1]*(dims[2]/2 + 1), NULL, 1, dims[2]*dims[1]*dims[0], CUFFT_Z2D, howmany);

  return out;
}

void destroy_plans(array<cufftHandle,2>& plans){
  cufftDestroy(plans[0]);
  cufftDestroy(plans[1]);
}
#endif

template<>
fft3d<2>::fft3d(array<Index,3> dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq) {
  if(real.sl == stloc::host) {
    plans = create_plans_3d(dims_, real, freq);
  } else {
    #ifdef __CUDACC__
    cuda_plans = create_plans_3d(dims_, real.shape()[1]);
    #endif
  }
}

template<>
fft3d<1>::fft3d(array<Index,3> dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq) {
  if(real.sl == stloc::host) {
    plans = create_plans_3d(dims_, real, freq);
  } else {
    #ifdef __CUDACC__
    cuda_plans = create_plans_3d(dims_, 1);
    #endif
  }
}

template<size_t d>
void fft3d<d>::forward(multi_array<double,d>& real, multi_array<complex<double>,d>& freq) {
  if(real.sl == stloc::host) {
    fftw_execute_dft_r2c(plans[0],real.data(),(fftw_complex*)freq.data());
  } else {
    #ifdef __CUDACC__
    cufftExecD2Z(cuda_plans[0],real.data(),(cufftDoubleComplex*)freq.data());
    #endif
  }
}
template void fft3d<1>::forward(multi_array<double,1>& real, multi_array<complex<double>,1>& freq);
template void fft3d<2>::forward(multi_array<double,2>& real, multi_array<complex<double>,2>& freq);


template<size_t d>
void fft3d<d>::backward(multi_array<complex<double>,d>& freq, multi_array<double,d>& real) {
  if(real.sl == stloc::host) {
    fftw_execute_dft_c2r(plans[1],(fftw_complex*)freq.data(),real.data());
  } else {
    #ifdef __CUDACC__
    cufftExecZ2D(cuda_plans[1],(cufftDoubleComplex*)freq.data(),real.data());
    #endif
  }
}
template void fft3d<1>::backward(multi_array<complex<double>,1>& freq, multi_array<double,1>& real);
template void fft3d<2>::backward(multi_array<complex<double>,2>& freq, multi_array<double,2>& real);

