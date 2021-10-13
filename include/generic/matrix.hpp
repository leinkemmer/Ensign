#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>
#include <generic/kernels.hpp>

#ifndef __MKL__
extern "C" {
  extern int dgees_(char*,char*,void*,int*,double*,int*, int*, double*, double*, double*, int*, double*, int*, bool*,int*);
}
#endif

#ifdef __CUDACC__
//array<cufftHandle,2> create_plans_1d(Index dims_);
array<cufftHandle,2> create_plans_1d(Index dims_, int howmany);
//array<cufftHandle,2> create_plans_2d(array<Index,2> dims_);
array<cufftHandle,2> create_plans_2d(array<Index,2> dims_, int howmany);
//array<cufftHandle,2> create_plans_3d(array<Index,3> dims_);
array<cufftHandle,2> create_plans_3d(array<Index,3> dims_, int howmany);
void destroy_plans(array<cufftHandle,2>& plans);
#endif

template<class T>
void set_zero(multi_array<T,2>& a);

template<class T>
void set_identity(multi_array<T,2>& a);

template<class T>
void set_const(multi_array<T,1>& a, T alpha);

template<class T>
void set_const2(multi_array<T,2>& a, T alpha);

template<class T>
void ptw_mult_row(const multi_array<T,2>& a, const T* w, multi_array<T,2>& out);

template<class T>
void transpose_inplace(multi_array<T,2>& a);

template<class T>
void matmul(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

template<class T>
void matmul_transa(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

template<class T>
void matmul_transa(const T* a, const T* b, T* c);

template<class T>
void matmul_transb(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

template<class T>
void matmul_transab(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

template<class T>
void matvec(const multi_array<T,2>& a, const multi_array<T,1>& b, multi_array<T,1>& c);

template<class T>
void matvec_trans(const multi_array<T,2>& a, const multi_array<T,1>& b, multi_array<T,1>& c);


array<fftw_plan,2> create_plans_1d(Index dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq);
array<fftw_plan,2> create_plans_1d(Index dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq);

array<fftw_plan,2> create_plans_2d(array<Index,2> dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq);
array<fftw_plan,2> create_plans_2d(array<Index,2> dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq);

array<fftw_plan,2> create_plans_3d(array<Index,3> dims_, multi_array<double,2>& real, multi_array<complex<double>,2>& freq);
array<fftw_plan,2> create_plans_3d(array<Index,3> dims_, multi_array<double,1>& real, multi_array<complex<double>,1>& freq);

void destroy_plans(array<fftw_plan,2>& plans);

#ifdef __MKL__
void schur(multi_array<double,2>& CC, multi_array<double,2>& TT, multi_array<double,1>& diag_r, MKL_INT& lwork);
#else
void schur(multi_array<double,2>& CC, multi_array<double,2>& TT, multi_array<double,1>& diag_r, int& lwork);
#endif
//template<class T>
//void transpose(const multi_array<T,2>& a, multi_array<T,2>& b);
