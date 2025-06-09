#include <generic/common.hpp>

#ifdef __CUDACC__

namespace Ensign {

template<class T>
__global__ void fill_gpu(int n, T* v, T alpha);

template<class T>
__global__ void ptw_mult_scal(int n, T* A, T alpha);

__global__ void ptw_mult_scal_cplx(int n, cuDoubleComplex* A, cuDoubleComplex alpha);

template<class T>
__global__ void ptw_mult_row_k(int nm, int n, T* A, T* v, T* B);

__global__ void ptw_mult_row_k(int nm, int n, cuDoubleComplex* A, cuDoubleComplex* v, cuDoubleComplex* B);

template<class T>
__global__ void ptw_sum_scal(int n, T* A, T alpha);

template<class T>
__global__ void ptw_sum(int n, T* A, T* B);

__global__ void dmaxpy(int n, double* a, double* x, double* y);

__global__ void scale_unique(double* x, double alpha);

__global__ void scale_sqrt_unique(double* x, double alpha);

__global__ void ptw_div_gs(int n, double* A, double* alpha);

__global__ void ptw_mult(int n, double* A, double* B, double* C);

__global__ void der_fourier(int n, cuDoubleComplex* A, double ax, double bx, int nx);

__global__ void ptw_mult_row_cplx_fourier(int nm, int n, int nx, cuDoubleComplex* A, double ax, double bx);

__global__ void cplx_conv(int n, double* A, cuDoubleComplex* B);

__device__ cuDoubleComplex expim(double z);

__device__ cuDoubleComplex phi1im(double z);

__device__ cuDoubleComplex phi2im(double z);

__global__ void exp_euler_fourier(int nm, int n, cuDoubleComplex* A, double* dc_r, double ts, cuDoubleComplex* T, double ax, double bx);

__global__ void second_ord_stage_fourier(int nm, int n, cuDoubleComplex* A, double* dc_r, double ts, cuDoubleComplex* T, cuDoubleComplex* U, double ax, double bx);

__global__ void ptw_mult_cplx(int n, cuDoubleComplex* A, double alpha);

__global__ void ptw_mult_scal(int n, double* A, double alpha, double* B);

__global__ void expl_euler(int n, double* A, double t, double* M1, double* M2);

__global__ void rk4(int n, double* A, double t, double* M1, double* M2, double* M3);

__global__ void rk4_finalcomb(int n, double* A, double t, double* M1, double* M2, double* M3, double* M4, double* M5);

__global__ void transpose_inplace(int n, double* A);

__global__ void der_fourier_2d(int N, int nx, int ny, cuDoubleComplex* A, double* lims, double nxx, cuDoubleComplex* B,cuDoubleComplex* C);

__global__ void ptw_mult_row_cplx_fourier_2d(int N, int nx, int ny, cuDoubleComplex* A, double* lims, double nxx, cuDoubleComplex* B, cuDoubleComplex* C); // Very similar, maybe can be put together

__global__ void exact_sol_exp_2d(int N, int nx, int ny, cuDoubleComplex* A, double* dc_r, double ts, double* lims, double nxx); // Very similar, maybe can be put together

__global__ void ptw_sum_complex(int n, cuDoubleComplex* A, cuDoubleComplex* B);

__global__ void ptw_sum_3mat(int n, double* A, double* B, double* C);

template<class T>
__global__ void ptw_diff(int n, T* A, T* B);

__global__ void exp_euler_fourier_2d(int N, int nx, int ny, cuDoubleComplex* A, double* dc_r, double ts, double* lims, cuDoubleComplex* T); // Very similar, maybe can be put together

__global__ void second_ord_stage_fourier_2d(int N, int nx, int ny, cuDoubleComplex* A, double* dc_r, double ts, double* lims, cuDoubleComplex* T, cuDoubleComplex* U); // Very similar, maybe can be put together

__global__ void der_fourier_3d(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* lims, double nxx, cuDoubleComplex* B, cuDoubleComplex* C, cuDoubleComplex* D);

__global__ void ptw_mult_row_cplx_fourier_3d(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* lims, double nxx, cuDoubleComplex* B, cuDoubleComplex* C, cuDoubleComplex* D); // Very similar, maybe can be put together

__global__ void exact_sol_exp_3d_a(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* dc_r, double ts, double* lims); // Very similar, maybe can be put together

__global__ void exact_sol_exp_3d_b(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* dc_r, double ts, double* lims, double nxx); // Very similar, maybe can be put together

__global__ void exact_sol_exp_3d_c(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* dc_r, double ts, double* lims); // Very similar, maybe can be put together

__global__ void exact_sol_exp_3d_d(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* dc_r, double ts, double* lims, double nxx);

__global__ void exp_euler_fourier_3d(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* dc_r, double ts, double* lims, cuDoubleComplex* T);

__global__ void second_ord_stage_fourier_3d(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* dc_r, double ts, double* lims, cuDoubleComplex* T, cuDoubleComplex* U);

template<class T>
__global__ void copy_R(int m, int n, T* Q, T* R, T w);

template<class T>
__global__ void div_Q(int m, int n, T* Q, T w);

} // namespace Ensign

#endif
