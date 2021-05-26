#include <generic/kernels.hpp>

#ifdef __CUDACC__

template<class T>
__global__ void fill_gpu(int n, T* v, T alpha){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    v[idx] = alpha;
    idx += blockDim.x * gridDim.x;
  }
}
template __global__ void fill_gpu(int n, double*, double);
template __global__ void fill_gpu(int n, float*, float);

template<class T>
__global__ void ptw_mult_scal(int n, T* A, T alpha){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    A[idx] *= alpha;
    idx += blockDim.x * gridDim.x;
  }
}
template __global__ void ptw_mult_scal(int, double*, double);
template __global__ void ptw_mult_scal(int, float*, float);

template<class T>
__global__ void ptw_mult_row_k(int nm, int n, T* A, T* v, T* B){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < nm){
    B[idx] = A[idx] * v[idx % n];
    idx += blockDim.x * gridDim.x;
  }
}
template __global__ void ptw_mult_row_k(int, int, double*, double*, double*);
template __global__ void ptw_mult_row_k(int, int, float*, float*, float*);

template<class T>
__global__ void ptw_sum(int n, T* A, T* B){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    A[idx] += B[idx];
    idx += blockDim.x * gridDim.x;
  }
}
template __global__ void ptw_sum(int, double*, double*);
template __global__ void ptw_sum(int, float*, float*);


template<class T>
__global__ void ptw_sum_scal(int n, T* A, T alpha){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    A[idx] += alpha;
    idx += blockDim.x * gridDim.x;
  }
}
template __global__ void ptw_sum_scal(int, double*, double);
template __global__ void ptw_sum_scal(int, float*, float);

__global__ void der_fourier(int n, cuDoubleComplex* A, double ax, double bx, int nx){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(idx == 0){
    A[idx] = make_cuDoubleComplex(0.0, 0.0);
    idx += blockDim.x * gridDim.x;
  }
  while(idx < n){
    cuDoubleComplex c = make_cuDoubleComplex(0.0, (2.0*M_PI/(bx-ax))*idx*nx);
    A[idx] = cuCdiv(A[idx],c);
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void ptw_mult_row_cplx_fourier(int nm, int n, int nx, cuDoubleComplex* A, double ax, double bx){ // Very similar, maybe can be put together
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < nm){
    cuDoubleComplex c = make_cuDoubleComplex(0.0, 2.0*M_PI/(nx*(bx-ax))*(idx%n));
    A[idx] = cuCmul(A[idx],c);
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void cplx_conv(int n, double* A, cuDoubleComplex* B){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    B[idx] = make_cuDoubleComplex(A[idx], 0.0);
    idx += blockDim.x * gridDim.x;
  }
}

__device__ cuDoubleComplex expim(double z){
    cuDoubleComplex out;
    sincos(z, &out.y, &out.x);

    return out;
}

__device__ cuDoubleComplex phi1im(double z){

    cuDoubleComplex out;

    if(abs(z) < 1e-7){
      out.x = 1.0 + z;
      out.y = 0.0;
    }else{
      out.x = sin(z)/z;
      out.y = 2*(pow(sin(z/2.0),2))/z;
    }

    return out;

}

__global__ void exp_euler_fourier(int nm, int n, cuDoubleComplex* A, double* dc_r, double ts, cuDoubleComplex* T, double ax, double bx){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  cuDoubleComplex tt = make_cuDoubleComplex(ts, 0.0);

  while(idx < nm){
    A[idx] = cuCmul(A[idx],expim(-ts*2.0*M_PI/(bx-ax)*(idx%n)*dc_r[idx / n]));
    A[idx] = cuCadd(A[idx],cuCmul(tt,cuCmul(phi1im(-ts*2.0*M_PI/(bx-ax)*(idx%n)*dc_r[idx / n]),T[idx])));
    idx += blockDim.x * gridDim.x;
  }

}

__global__ void ptw_mult_cplx(int n, cuDoubleComplex* A, double alpha){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  cuDoubleComplex alphac = make_cuDoubleComplex(alpha, 0.0);

  while(idx < n){
    A[idx] = cuCmul(alphac,A[idx]);
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void ptw_mult_scal(int n, double* A, double alpha, double* B){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    B[idx] = alpha*A[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void expl_euler(int n, double* A, double t, double* M1, double* M2){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    A[idx] += t*(M1[idx]-M2[idx]);
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void transpose_inplace(int n, double* A){

  int i = blockIdx.x % n ; // n number of rows
  int j = blockIdx.x / n;
  double tmp;
  if((i < n) && (j < i)){
    tmp = A[i+j*n];
    A[i+j*n] = A[j+i*n];
    A[j+i*n] = tmp;
  }
}

__global__ void der_fourier_2d(int N, int nx, int ny, cuDoubleComplex* A, double* lims, double nxx, cuDoubleComplex* B,cuDoubleComplex* C){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < N){
    int i = idx % nx;
    int j = idx / nx;

    if(j==ny/2){
      j = 0;
    }else if(j > ny/2){
      j -= ny;
    }

    cuDoubleComplex c1 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[1]-lims[0]))*i);
    cuDoubleComplex c2 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[3]-lims[2]))*j);

    cuDoubleComplex mm = cuCdiv(cuCadd(cuCmul(c1,c1),cuCmul(c2,c2)),make_cuDoubleComplex(nxx,0.0));

    if((i == 0) && (j == 0)){
      B[idx] = make_cuDoubleComplex(0.0, 0.0);
      C[idx] = make_cuDoubleComplex(0.0, 0.0);
    }else{
      B[idx] = cuCmul(A[idx],cuCdiv(c1,mm));
      C[idx] = cuCmul(A[idx],cuCdiv(c2,mm));
    }
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void ptw_mult_row_cplx_fourier_2d(int N, int nx, int ny, cuDoubleComplex* A, double* lims, double nxx, cuDoubleComplex* B, cuDoubleComplex* C){ // Very similar, maybe can be put together
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < N){
    int i = (idx % (nx*ny)) % nx;
    int j = (idx % (nx*ny)) / nx;

    if(j==ny/2){
      j = 0;
    }else if(j > ny/2){
      j -= ny;
    }

    cuDoubleComplex c1 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[1]-lims[0]))*i*nxx);
    cuDoubleComplex c2 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[3]-lims[2]))*j*nxx);

    B[idx] = cuCmul(A[idx],c1);
    C[idx] = cuCmul(A[idx],c2);

    idx += blockDim.x * gridDim.x;
  }

}

__global__ void exact_sol_exp_2d(int N, int nx, int ny, cuDoubleComplex* A, double* dc_r, double ts, double* lims, double nxx){ // Very similar, maybe can be put together
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  cuDoubleComplex nxx_c = make_cuDoubleComplex(nxx, 0.0);

  while(idx < N){
    int i = (idx % (nx*ny)) % nx;

    A[idx] = cuCmul(A[idx],cuCmul(expim(-ts*2.0*M_PI/(lims[1]-lims[0])*i*dc_r[idx / (nx*ny)]),nxx_c));

    idx += blockDim.x * gridDim.x;
  }

}

__global__ void ptw_sum_complex(int n, cuDoubleComplex* A, cuDoubleComplex* B){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    A[idx] = cuCadd(A[idx],B[idx]);
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void ptw_sum_3mat(int n, double* A, double* B, double* C){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    A[idx] += B[idx];
    A[idx] += C[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void ptw_diff(int n, double* A, double* B){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < n){
    A[idx] -= B[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void exp_euler_fourier_2d(int N, int nx, int ny, cuDoubleComplex* A, double* dc_r, double ts, double* lims, cuDoubleComplex* T){ // Very similar, maybe can be put together
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  cuDoubleComplex tt = make_cuDoubleComplex(ts, 0.0);

  while(idx < N){
    int j = (idx % (nx*ny)) / nx;

    if(j==ny/2){
      j = 0;
    }else if(j > ny/2){
      j -= ny;
    }

    A[idx] = cuCmul(A[idx],expim(-ts*(2.0*M_PI/(lims[3]-lims[2]))*j*dc_r[idx / (nx*ny)]));
    A[idx] = cuCadd(A[idx],cuCmul(tt,cuCmul(phi1im(-ts*(2.0*M_PI/(lims[3]-lims[2]))*j*dc_r[idx / (nx*ny)]),T[idx])));

    idx += blockDim.x * gridDim.x;
  }

}

__global__ void der_fourier_3d(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* lims, double nxx, cuDoubleComplex* B, cuDoubleComplex* C, cuDoubleComplex* D){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < N){
    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = idx / (nx*ny);

    if(j==ny/2){
      j = 0;
    }else if(j > ny/2){
      j -= ny;
    }

    if(k==nz/2){
      k = 0;
    }else if(k > nz/2){
      k -= nz;
    }

    cuDoubleComplex c1 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[1]-lims[0]))*i);
    cuDoubleComplex c2 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[3]-lims[2]))*j);
    cuDoubleComplex c3 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[5]-lims[4]))*k);

    cuDoubleComplex mm = cuCdiv(cuCadd(cuCadd(cuCmul(c1,c1),cuCmul(c2,c2)),cuCmul(c3,c3)),make_cuDoubleComplex(nxx,0.0));

    if((i == 0) && (j == 0) && (k == 0)){
      B[idx] = make_cuDoubleComplex(0.0, 0.0);
      C[idx] = make_cuDoubleComplex(0.0, 0.0);
      D[idx] = make_cuDoubleComplex(0.0, 0.0);
    }else{
      B[idx] = cuCmul(A[idx],cuCdiv(c1,mm));
      C[idx] = cuCmul(A[idx],cuCdiv(c2,mm));
      D[idx] = cuCmul(A[idx],cuCdiv(c3,mm));
    }
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void ptw_mult_row_cplx_fourier_3d(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* lims, double nxx, cuDoubleComplex* B, cuDoubleComplex* C, cuDoubleComplex* D){ // Very similar, maybe can be put together
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < N){
    int i = (idx % (nx*ny*nz)) % nx;
    int j = ((idx % (nx*ny*nz)) / nx) % ny;
    int k = (idx % (nx*ny*nz)) / (nx*ny);

    if(j==ny/2){
      j = 0;
    }else if(j > ny/2){
      j -= ny;
    }

    if(k==nz/2){
      k = 0;
    }else if(k > nz/2){
      k -= nz;
    }

    cuDoubleComplex c1 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[1]-lims[0]))*i*nxx);
    cuDoubleComplex c2 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[3]-lims[2]))*j*nxx);
    cuDoubleComplex c3 = make_cuDoubleComplex(0.0, (2.0*M_PI/(lims[5]-lims[4]))*k*nxx);

    B[idx] = cuCmul(A[idx],c1);
    C[idx] = cuCmul(A[idx],c2);
    D[idx] = cuCmul(A[idx],c3);

    idx += blockDim.x * gridDim.x;
  }

}

__global__ void exact_sol_exp_3d_a(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* dc_r, double ts, double* lims){ // Very similar, maybe can be put together
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while(idx < N){
    int i = (idx % (nx*ny*nz)) % nx;

    A[idx] = cuCmul(A[idx],expim(-ts*2.0*M_PI/(lims[1]-lims[0])*i*dc_r[idx / (nx*ny*nz)]));

    idx += blockDim.x * gridDim.x;
  }

}

__global__ void exact_sol_exp_3d_b(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* dc_r, double ts, double* lims, double nxx){ // Very similar, maybe can be put together
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  cuDoubleComplex nxx_c = make_cuDoubleComplex(nxx, 0.0);

  while(idx < N){
    int j = ((idx % (nx*ny*nz)) / nx) % ny;

    if(j==ny/2){
      j = 0;
    }else if(j > ny/2){
      j -= ny;
    }


    A[idx] = cuCmul(A[idx],cuCmul(expim(-ts*2.0*M_PI/(lims[3]-lims[2])*j*dc_r[idx / (nx*ny*nz)]),nxx_c));

    idx += blockDim.x * gridDim.x;
  }

}

__global__ void exp_euler_fourier_3d(int N, int nx, int ny, int nz, cuDoubleComplex* A, double* dc_r, double ts, double* lims, cuDoubleComplex* T){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  cuDoubleComplex tt = make_cuDoubleComplex(ts, 0.0);

  while(idx < N){
    int k = (idx % (nx*ny*nz)) / (nx*ny);

    if(k==nz/2){
      k = 0;
    }else if(k > nz/2){
      k -= nz;
    }

    A[idx] = cuCmul(A[idx],expim(-ts*(2.0*M_PI/(lims[5]-lims[4]))*k*dc_r[idx / (nx*ny*nz)]));
    A[idx] = cuCadd(A[idx],cuCmul(tt,cuCmul(phi1im(-ts*(2.0*M_PI/(lims[5]-lims[4]))*k*dc_r[idx / (nx*ny*nz)]),T[idx])));

    idx += blockDim.x * gridDim.x;
  }

}

#endif
