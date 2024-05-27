#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/kernels.hpp>
#include <generic/timer.hpp>
#include <generic/fft.hpp>
#include <generic/netcdf.hpp>
#include <generic/kernels.hpp>

#include <cxxopts.hpp>

bool CPU;

template<size_t d> using mind = array<Index,d>;
template<size_t d> using mfp  = array<double,d>;
using mat  = multi_array<double,2>;
using vec  = multi_array<double,1>;
using cmat = multi_array<complex<double>,2>;
using cvec = multi_array<complex<double>,1>;


Index freq(Index k, Index n) {
  if(k < n/2)
    return k;
  else if(k == n/2)
    return 0;
  else
    return k-n;
}

template<size_t d>
struct grid_info {
  Index r;
  mind<d>  N_xx, N_vv;
  mfp<2*d> lim_xx, lim_vv;
  mfp<d>   h_xx, h_vv;
  Index dxx_mult, dvv_mult, dxxh_mult, dvvh_mult;

  grid_info(Index _r, mind<d> _N_xx, mind<d> _N_vv, mfp<2*d> _lim_xx, mfp<2*d> _lim_vv)
    : r(_r), N_xx(_N_xx), N_vv(_N_vv), lim_xx(_lim_xx), lim_vv(_lim_vv) {

    // compute h_xx and h_vv
    for(int ii = 0; ii < 3; ii++){
      Index jj = 2*ii;
      h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
      h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
    }
  
    dxx_mult  = N_xx[0]*N_xx[1]*N_xx[2];
    dvv_mult  = N_vv[0]*N_vv[1]*N_vv[2];
    dxxh_mult = N_xx[2]*N_xx[1]*(N_xx[0]/2 + 1);
    dvvh_mult = N_vv[2]*N_vv[1]*(N_vv[0]/2 + 1);
  }

  Index lin_idx_x(mind<d> i) const {
    Index idx=0, stride=1;
    for(size_t k=0;k<d;k++) {
      idx += stride*i[k];
      stride *= N_xx[k];
    }
    return idx;
  }
  
  Index lin_idx_v(mind<d> i) const {
    Index idx=0, stride=1;
    for(size_t k=0;k<d;k++) {
      idx += stride*i[k];
      stride *= N_vv[k];
    }
    return idx;
  }

  double x(size_t k, Index i) const {
    return lim_xx[2*k] + i*h_xx[k];
  }

  mfp<d> x(mind<d> i) const {
    mfp<d> out;
    for(size_t k=0;k<d;k++)
      out[k] = x(k, i[k]);
    return out;
  }

  double v(size_t k, Index i) const {
    return lim_vv[2*k] + i*h_vv[k];
  }

  mfp<d> v(mind<d> i) const {
    mfp<d> out;
    for(size_t k=0;k<d;k++)
      out[k] = v(k, i[k]);
    return out;
  }
};

array<mat,3> create_mat_array(mind<2> dim, stloc sl) {
  return {mat(dim,sl), mat(dim,sl), mat(dim,sl)};
}

array<cmat,3> create_cmat_array(mind<2> dim, stloc sl) {
  return {cmat(dim,sl), cmat(dim,sl), cmat(dim,sl)};
}

array<vec,3> create_vec_array(Index dim, stloc sl) {
  array<vec,3> out = {vec({dim}, sl), vec({dim}, sl), vec({dim}, sl)};
  return out;
  //return {vec({dim}, sl), vec({dim}, sl), vec({dim}, sl)};
}

array<cvec,3> create_cvec_array(Index dim, stloc sl) {
  return {cvec({dim}, sl), cvec({dim}, sl), cvec({dim}, sl)};
}

// Note that using a std::function object has a non-negligible performance overhead
template<class func>
void componentwise_vec_omp(const mind<3>& N, func F) {
  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index k = 0; k < N[2]; k++){
    for(Index j = 0; j < N[1]; j++){
      for(Index i = 0; i < N[0]; i++){
        Index idx = i+j*N[0] + k*(N[0]*N[1]);
        F(idx, {i,j,k});
      }
    }
  }
}


template<class func>
void componentwise_vec_fourier_omp(const mind<3>& N, func F) {
  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index k = 0; k < N[2]; k++){
    for(Index j = 0; j < N[1]; j++){
      for(Index i = 0; i < (N[0]/2+1); i++){
        Index idx = i+j*(N[0]/2+1) + k*((N[0]/2+1)*N[1]);
        F(idx, {i,j,k});
      }
    }
  }
}

template<class func>
void componentwise_mat_fourier_omp(Index r, const mind<3>& N,  func F) {
  #ifdef __OPENMP__
  #pragma omp parallel for collapse(2)
  #endif
  for(int rr = 0; rr < r; rr++){
    for(Index k = 0; k < N[2]; k++){
      for(Index j = 0; j < N[1]; j++){
        for(Index i = 0; i < (N[0]/2 + 1); i++){
          Index idx = i+j*(N[0]/2+1) + k*((N[0]/2+1)*N[1]);
          F(idx, {i,j,k}, rr);
        }
      }
    }
  }
}



void diagonalize_coeff(const array<mat,3>& C1, array<cmat,3>& Tc, array<vec,3>& dc_r, const grid_info<3>& gi, diagonalization& schur) {

  array<mat,3> T = create_mat_array({gi.r, gi.r}, stloc::host);

  if(C1[0].sl == stloc::host) {
    schur(C1[0], T[0], dc_r[0]);
    schur(C1[1], T[1], dc_r[1]);
    schur(C1[2], T[2], dc_r[2]);

    T[0].to_cplx(Tc[0]);
    T[1].to_cplx(Tc[1]);
    T[2].to_cplx(Tc[2]);
  } else {
    array<mat,3> h_C1 = create_mat_array({gi.r,gi.r}, stloc::host);
    array<vec,3> h_dc_r = create_vec_array(gi.r, stloc::host);

    h_C1[0] = C1[0];
    h_C1[1] = C1[1];
    h_C1[2] = C1[2];

    schur(h_C1[0], T[0], h_dc_r[0]);
    schur(h_C1[1], T[1], h_dc_r[1]);
    schur(h_C1[2], T[2], h_dc_r[2]);

    dc_r[0] = h_dc_r[0];
    dc_r[1] = h_dc_r[1];
    dc_r[2] = h_dc_r[2];

    array<cmat,3> h_Tc = create_cmat_array({gi.r,gi.r}, stloc::host);
    T[0].to_cplx(h_Tc[0]);
    T[1].to_cplx(h_Tc[1]);
    T[2].to_cplx(h_Tc[2]);

    Tc[0] = h_Tc[0];
    Tc[1] = h_Tc[1];
    Tc[2] = h_Tc[2];
  }
}


struct PS_K_step {

  void internal_step_1(double tau) {
    blas->matmul(Khat,Tc[0],Mhat);

    if(sl == stloc::host) {
      componentwise_mat_fourier_omp(gi.r, gi.N_xx, [this, tau](Index idx, mind<3> i, Index rr) {
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[1]-gi.lim_xx[0])*i[0]);
        Mhat(idx,rr) *= exp(-tau*lambdax*dc_r[0](rr));
      });
    } else {
      #ifdef __CUDACC__
      exact_sol_exp_3d_a<<<(Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Mhat.num_elements(), gi.N_xx[0]/2 + 1, gi.N_xx[1], gi.N_xx[2], (cuDoubleComplex*)Mhat.begin(), dc_r[0].begin(), tau, d_lim_xx->data());
      #endif
    }
    blas->matmul_transb(Mhat,Tc[0],Khat);
  }

  void internal_step_2(double tau) {
    double ncxx = 1.0 / (gi.N_xx[0]*gi.N_xx[1]*gi.N_xx[2]);

    // Second internal splitting step
    blas->matmul(Khat,Tc[1],Mhat);

    if(Khat.sl == stloc::host) {
      componentwise_mat_fourier_omp(gi.r, gi.N_xx, [this, tau, ncxx](Index idx, mind<3> i, Index rr) {
        Index mult_j = freq(i[1], gi.N_xx[1]);
        complex<double> lambday = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[3]-gi.lim_xx[2])*mult_j);
        Mhat(idx,rr) *= exp(-tau*lambday*dc_r[1](rr))*ncxx;
      });
    } else {
      #ifdef __CUDACC__
      exact_sol_exp_3d_b<<<(Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Mhat.num_elements(), gi.N_xx[0]/2 + 1, gi.N_xx[1], gi.N_xx[2], (cuDoubleComplex*)Mhat.begin(), dc_r[1].begin(), tau, d_lim_xx->data(), ncxx);
      #endif
    }
    blas->matmul_transb(Mhat,Tc[1],Khat);
  }

  void internal_step_3_stage1(double tau, mat& K, array<vec,3>& ef, const array<cmat,3>& C2c) {
    ptw_mult_row(K,ef[0],Ke[0]);
    ptw_mult_row(K,ef[1],Ke[1]);
    ptw_mult_row(K,ef[2],Ke[2]);

    fft->forward(Ke[0], Kehat[0]);
    fft->forward(Ke[1], Kehat[1]);
    fft->forward(Ke[2], Kehat[2]);

    blas->matmul_transb(Kehat[0],C2c[0],Khat);
    blas->matmul_transb(Kehat[1],C2c[1],tmpXhat);

    Khat += tmpXhat;

    blas->matmul_transb(Kehat[2],C2c[2],tmpXhat);

    Khat += tmpXhat;

    blas->matmul(Khat,Tc[2],tmpXhat);

    if(K.sl == stloc::host) {
      componentwise_mat_fourier_omp(gi.r, gi.N_xx, [this, tau](Index idx, mind<3> i, Index rr) {
        Index mult_k = freq(i[2], gi.N_xx[2]);
        complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[5]-gi.lim_xx[4])*mult_k);
        Mhat(idx,rr) *= exp(-tau*lambdaz*dc_r[2](rr));
        Mhat(idx,rr) += tau*phi1_im(-tau*lambdaz*dc_r[2](rr))*tmpXhat(idx,rr);
      });
    } else {
        #ifdef __CUDACC__
        exp_euler_fourier_3d<<<(Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Mhat.num_elements(), gi.N_xx[0]/2 + 1, gi.N_xx[1], gi.N_xx[2], (cuDoubleComplex*)Mhat.begin(),dc_r[2].begin(),tau, d_lim_xx->data(), (cuDoubleComplex*)tmpXhat.begin());
        #endif
    }

    blas->matmul_transb(Mhat,Tc[2],Khat);
    
    double ncxx = 1.0 / (gi.dxx_mult);

    if(Khat.sl == stloc::host) { // TODO
      Khat *= ncxx;
    } else {
      #ifdef __CUDACC__
      ptw_mult_cplx<<<(Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Khat.num_elements(), (cuDoubleComplex*)Khat.begin(), ncxx);
      #endif
    }

    fft->backward(Khat, K);
  }

  void internal_step_3_stage2(double tau, mat& K, array<vec,3>& ef, const array<cmat,3>& C2c) {
    ptw_mult_row(K,ef[0],Ke[0]);
    ptw_mult_row(K,ef[1],Ke[1]);
    ptw_mult_row(K,ef[2],Ke[2]);

    fft->forward(Ke[0], Kehat[0]);
    fft->forward(Ke[1], Kehat[1]);
    fft->forward(Ke[2], Kehat[2]);

    blas->matmul_transb(Kehat[0],C2c[0],Khat);
    blas->matmul_transb(Kehat[1],C2c[1],Kehat[0]);

    Kehat[0] += Khat;

    blas->matmul_transb(Kehat[2],C2c[2],Khat);

    Kehat[0] += Khat;

    blas->matmul(Kehat[0],Tc[2],Khat);

    if(K.sl == stloc::host) {
      // TODO: this is missing in the GPU code
      Khat -= tmpXhat;

      componentwise_mat_fourier_omp(gi.r, gi.N_xx, [this, tau](Index idx, mind<3> i, Index rr) {
        Index mult_k = freq(i[2], gi.N_xx[2]);
        complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[5]-gi.lim_xx[4])*mult_k);
        Mhat(idx,rr) += tau*phi2_im(-tau*lambdaz*dc_r[2](rr))*Khat(idx,rr);
      });
    } else {
        #ifdef __CUDACC__
        second_ord_stage_fourier_3d<<<(Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Mhat.num_elements(), gi.N_xx[0]/2 + 1, gi.N_xx[1], gi.N_xx[2], (cuDoubleComplex*)Mhat.begin(),dc_r[2].begin(),tau, d_lim_xx->data(), (cuDoubleComplex*)tmpXhat.begin(), (cuDoubleComplex*)Khat.begin());
        #endif
    }

    blas->matmul_transb(Mhat,Tc[2],Khat);

    double ncxx = 1.0 / (gi.dxx_mult);
    if(K.sl == stloc::host)
      Khat *= ncxx;
    else {
      #ifdef __CUDACC__
      ptw_mult_cplx<<<(Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Khat.num_elements(), (cuDoubleComplex*)Khat.begin(), ncxx);
      #endif
    }

    fft->backward(Khat, K);
  }

  void operator()(double tau, mat& K, array<vec,3>& ef, const array<mat,3>& C1, const array<cmat,3>& C2c, Index nsteps_ei=1) {

    if(fft == nullptr)
      fft = make_unique_ptr<fft3d<2>>(gi.N_xx, K, Khat);

    diagonalize_coeff(C1, Tc, dc_r, gi, schur);

    // Transform to Fourier space
    fft->forward(K, Khat);

    internal_step_1(tau);
    internal_step_2(tau);

    // Inverse Fourier transform
    fft->backward(Khat, K);

    fft->forward(K, Khat); // TODO
    blas->matmul(Khat,Tc[2],Mhat);
    for(Index jj = 0; jj < nsteps_ei; jj++){
      internal_step_3_stage1(tau/double(nsteps_ei), K, ef, C2c);
      internal_step_3_stage2(tau/double(nsteps_ei), K, ef, C2c);
    }
  }

  PS_K_step(stloc _sl, grid_info<3> _gi, const blas_ops* _blas)
    : sl(_sl), gi(_gi), blas(_blas), fft(nullptr), Khat(_sl), Mhat(_sl), tmpXhat(_sl), schur(_gi.r) {

      Khat.resize({gi.dxxh_mult,gi.r});
      Mhat.resize({gi.dxxh_mult,gi.r});
      tmpXhat.resize({gi.dxxh_mult,gi.r});

      Ke = create_mat_array({gi.dxx_mult,gi.r}, sl);
      Kehat = create_cmat_array({gi.dxxh_mult,gi.r}, sl);
      Tc = create_cmat_array({gi.r,gi.r}, sl);
      dc_r = create_vec_array(gi.r, sl);

      #ifdef __CUDACC__
      if(sl == stloc::device) {
        d_lim_xx = make_unique_ptr<vec>(array<Index,1>({6}), stloc::device);
        cudaMemcpy(d_lim_xx->data(), gi.lim_xx.data(), 6*sizeof(double), cudaMemcpyHostToDevice);
      }
      #endif
    }


private:
  grid_info<3> gi;
  stloc sl;
  const blas_ops* blas;

  std::unique_ptr<fft3d<2>> fft;
  array<mat,3> Ke;
  array<cmat,3> Kehat;
  cmat Khat, Mhat, tmpXhat;
  diagonalization schur;
  array<cmat,3> Tc;
  array<vec,3> dc_r;
  std::unique_ptr<vec> d_lim_xx;
};


struct PS_L_step {

  void internal_step_1(double tau) {
      blas->matmul(Lhat,Tc[0],Nhat);

      if(Nhat.sl == stloc::host) {
        componentwise_mat_fourier_omp(gi.r, gi.N_vv, [this, tau](Index idx, mind<3> i, Index rr) {
          complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[1]-gi.lim_vv[0])*i[0]);
          Nhat(idx,rr) *= exp(tau*lambdav*dd1_r[0](rr));
        });
      } else {
        #ifdef __CUDACC__
        exact_sol_exp_3d_a<<<(Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Nhat.num_elements(), gi.N_vv[0]/2 + 1, gi.N_vv[1], gi.N_vv[2], (cuDoubleComplex*)Nhat.begin(), dd1_r[0].begin(), -tau, d_lim_vv->data());
        #endif
      }
      
      blas->matmul_transb(Nhat,Tc[0],Lhat);
  }

  void internal_step_2(double tau) {
      blas->matmul(Lhat,Tc[1],Nhat);

      double ncvv = 1.0/double(gi.dvv_mult);
      if(Nhat.sl == stloc::host) {
        componentwise_mat_fourier_omp(gi.r, gi.N_vv, [this, tau, ncvv](Index idx, mind<3> i, Index rr) {
          Index mult_j = freq(i[1], gi.N_vv[1]);
          complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[3]-gi.lim_vv[2])*mult_j);
          Nhat(idx,rr) *= exp(tau*lambdaw*dd1_r[1](rr))*ncvv;
        });
      } else {
        #ifdef __CUDACC__
        exact_sol_exp_3d_b<<<(Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Nhat.num_elements(), gi.N_vv[0]/2 + 1, gi.N_vv[1], gi.N_vv[2], (cuDoubleComplex*)Nhat.begin(), dd1_r[1].begin(), -tau, d_lim_vv->data(), ncvv);
        #endif
      }

      blas->matmul_transb(Nhat,Tc[1],Lhat);
  }

  void internal_step_3_stage1(double tau, mat& L, const array<cmat,3>& D2c) {

    ptw_mult_row(L,v[0],Lv[0]);
    ptw_mult_row(L,v[1],Lv[1]);
    ptw_mult_row(L,v[2],Lv[2]);
    /*
    // TODO: faster?
    ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
    ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());
    ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_u.begin(),d_Lu.begin());
    */

    fft->forward(Lv[0], Lvhat[0]);
    fft->forward(Lv[1], Lvhat[1]);
    fft->forward(Lv[2], Lvhat[2]);

    blas->matmul_transb(Lvhat[0],D2c[0],Lhat);

    blas->matmul_transb(Lvhat[1],D2c[1],tmpVhat);

    Lhat += tmpVhat;

    blas->matmul_transb(Lvhat[2],D2c[2],tmpVhat);

    Lhat += tmpVhat;

    blas->matmul(Lhat,Tc[2],tmpVhat);

    if(Nhat.sl == stloc::host) {
      componentwise_mat_fourier_omp(gi.r, gi.N_vv, [this, tau](Index idx, mind<3> i, Index rr) {
        Index mult_k = freq(i[2], gi.N_vv[2]);
        complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[5]-gi.lim_vv[4])*mult_k);
        Nhat(idx,rr) *= exp(tau*lambdau*dd1_r[2](rr));
        Nhat(idx,rr) -= tau*phi1_im(tau*lambdau*dd1_r[2](rr))*tmpVhat(idx,rr);
      });
    } else {
      #ifdef __CUDACC__
      exp_euler_fourier_3d<<<(Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Nhat.num_elements(), gi.N_vv[0]/2 + 1, gi.N_vv[1], gi.N_vv[2], (cuDoubleComplex*)Nhat.begin(),dd1_r[2].begin(),-tau, d_lim_vv->data(), (cuDoubleComplex*)tmpVhat.begin());
      #endif
    }

    blas->matmul_transb(Nhat,Tc[2],Lhat);

    double ncvv = 1.0/double(gi.dvv_mult);
    Lhat *= ncvv;

    fft->backward(Lhat, L);
  }
  
  void internal_step_3_stage2(double tau, mat& L, const array<cmat,3>& D2c) {

    ptw_mult_row(L,v[0],Lv[0]);
    ptw_mult_row(L,v[1],Lv[1]);
    ptw_mult_row(L,v[2],Lv[2]);

    fft->forward(Lv[0], Lvhat[0]);
    fft->forward(Lv[1], Lvhat[1]);
    fft->forward(Lv[2], Lvhat[2]);

    blas->matmul_transb(Lvhat[0],D2c[0],Lhat);
    blas->matmul_transb(Lvhat[1],D2c[1],Lvhat[0]);

    Lvhat[0] += Lhat;

    blas->matmul_transb(Lvhat[2],D2c[2],Lhat);

    Lvhat[0] += Lhat;

    blas->matmul(Lvhat[0],Tc[2],Lhat);

    if(Lhat.sl == stloc::host) {
      Lhat -= tmpVhat;

      componentwise_mat_fourier_omp(gi.r, gi.N_vv, [this, tau](Index idx, mind<3> i, Index rr) {
        Index mult_k = freq(i[2], gi.N_vv[2]);
        complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[5]-gi.lim_vv[4])*mult_k);
        Nhat(idx,rr) -= tau*phi2_im(tau*lambdau*dd1_r[2](rr))*Lhat(idx,rr);
      });
    } else {
      #ifdef __CUDACC__
      second_ord_stage_fourier_3d<<<(Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(Nhat.num_elements(), gi.N_vv[0]/2 + 1, gi.N_vv[1], gi.N_vv[2], (cuDoubleComplex*)Nhat.begin(),dd1_r[2].begin(),-tau, d_lim_vv->data(), (cuDoubleComplex*)tmpVhat.begin(), (cuDoubleComplex*)Lhat.begin());
      #endif
    }

    blas->matmul_transb(Nhat,Tc[2],Lhat);

    double ncvv = 1.0/double(gi.dvv_mult);
    Lhat *= ncvv;

    fft->backward(Lhat, L);
  }

  void operator()(double tau, mat& L, const array<mat,3>& D1, const array<mat,3>& D2, Index nsteps_ei=1) {

    if(fft == nullptr)
      fft = make_unique_ptr<fft3d<2>>(gi.N_vv, L, Lhat);

    diagonalize_coeff(D1, Tc, dd1_r, gi, schur);

    array<cmat,3> D2c = create_cmat_array({gi.r,gi.r}, L.sl);
    if(D2[0].sl == stloc::host) {
      D2[0].to_cplx(D2c[0]);
      D2[1].to_cplx(D2c[1]);
      D2[2].to_cplx(D2c[2]);
    } else {
      #ifdef __CUDACC__
      cplx_conv<<<(D2[0].num_elements()+n_threads-1)/n_threads,n_threads>>>(D2[0].num_elements(), D2[0].begin(), (cuDoubleComplex*)D2c[0].begin());
      cplx_conv<<<(D2[1].num_elements()+n_threads-1)/n_threads,n_threads>>>(D2[1].num_elements(), D2[1].begin(), (cuDoubleComplex*)D2c[1].begin());
      cplx_conv<<<(D2[2].num_elements()+n_threads-1)/n_threads,n_threads>>>(D2[2].num_elements(), D2[2].begin(), (cuDoubleComplex*)D2c[2].begin());
      #endif
    }

    fft->forward(L, Lhat);

    internal_step_1(tau);
    internal_step_2(tau);

    fft->backward(Lhat, L);

    fft->forward(L, Lhat); // TODO
    blas->matmul(Lhat,Tc[2],Nhat);
    for(Index jj = 0; jj < nsteps_ei; jj++) {
      internal_step_3_stage1(tau/double(nsteps_ei), L, D2c);
      internal_step_3_stage2(tau/double(nsteps_ei), L, D2c);
    }
  }
  
  PS_L_step(stloc sl, grid_info<3> _gi, const blas_ops* _blas)
    : gi(_gi), blas(_blas), Lhat(sl), Nhat(sl), tmpVhat(sl), schur(_gi.r) {
      
      Lhat.resize({gi.dvvh_mult,gi.r});
      Nhat.resize({gi.dvvh_mult,gi.r});
      tmpVhat.resize({gi.dvvh_mult,gi.r});

      Lv = create_mat_array({gi.dvv_mult,gi.r}, sl);
      Lvhat = create_cmat_array({gi.dvvh_mult,gi.r}, sl);
      Tc = create_cmat_array({gi.r,gi.r}, sl);
      v = create_vec_array(gi.dvv_mult, sl);
      dd1_r = create_vec_array(gi.r, sl);

      array<vec,3> h_v = create_vec_array(gi.dvv_mult, stloc::host);
      componentwise_vec_omp(gi.N_vv, [this, &h_v](Index idx, mind<3> i) {
        h_v[0](idx) = gi.v(0, i[0]);
        h_v[1](idx) = gi.v(1, i[1]);
        h_v[2](idx) = gi.v(2, i[2]);
      });
      v = h_v;

      #ifdef __CUDACC__
      if(sl == stloc::device) {
        d_lim_vv = make_unique_ptr<vec>(array<Index,1>({6}), stloc::device);
        cudaMemcpy(d_lim_vv->data(), gi.lim_vv.data(), 6*sizeof(double), cudaMemcpyHostToDevice);
      }
      #endif
  }

private:
  grid_info<3> gi;
  std::unique_ptr<fft3d<2>> fft;
  array<vec,3> v, dd1_r;
  array<mat,3> Lv;
  array<cmat,3> Tc;
  cmat Lhat, Nhat,  tmpVhat;
  array<cmat,3> Lvhat;
  std::unique_ptr<vec> d_lim_vv;
  diagonalization schur;
  const blas_ops* blas;
};



struct PS_S_step {

  PS_S_step(stloc sl, Index r)
    : tmpS(sl), tmpS1(sl), tmpS2(sl), tmpS3(sl), tmpS4(sl), Tv(sl), Tw(sl) {

      tmpS.resize({r,r});
      tmpS1.resize({r,r});
      tmpS2.resize({r,r});
      tmpS3.resize({r,r});
      tmpS4.resize({r,r});
      Tv.resize({r,r});
      Tw.resize({r,r});
  }

  void operator()(double tau, mat& S, const array<mat,3>& C1, const array<mat,3>& C2, const array<mat,3>& D1, const array<mat,3>& D2, const blas_ops& blas, Index nsteps_rk4=1) {

    double tau_rk4 = tau / double(nsteps_rk4);

    for(Index jj = 0; jj< nsteps_rk4; jj++){
      blas.matmul_transb(S,C1[0],tmpS);
      blas.matmul(D2[0],tmpS,tmpS1);
      blas.matmul_transb(S,C1[1],tmpS);
      blas.matmul(D2[1],tmpS,Tw);
      tmpS1 += Tw;
      blas.matmul_transb(S,C1[2],tmpS);
      blas.matmul(D2[2],tmpS,Tw);
      tmpS1 += Tw;
      blas.matmul_transb(S,C2[0],tmpS);
      blas.matmul(D1[0],tmpS,Tw);
      tmpS1 -= Tw;
      blas.matmul_transb(S,C2[1],tmpS);
      blas.matmul(D1[1],tmpS,Tw);
      tmpS1 -= Tw;
      blas.matmul_transb(S,C2[2],tmpS);
      blas.matmul(D1[2],tmpS,Tw);

      tmpS1 -= Tw;

      Tv = tmpS1;
      Tv *= (tau_rk4/2);
      Tv += S;

      blas.matmul_transb(Tv,C1[0],tmpS);
      blas.matmul(D2[0],tmpS,tmpS2);
      blas.matmul_transb(Tv,C1[1],tmpS);
      blas.matmul(D2[1],tmpS,Tw);
      tmpS2 += Tw;
      blas.matmul_transb(Tv,C1[2],tmpS);
      blas.matmul(D2[2],tmpS,Tw);
      tmpS2 += Tw;
      blas.matmul_transb(Tv,C2[0],tmpS);
      blas.matmul(D1[0],tmpS,Tw);
      tmpS2 -= Tw;
      blas.matmul_transb(Tv,C2[1],tmpS);
      blas.matmul(D1[1],tmpS,Tw);
      tmpS2 -= Tw;
      blas.matmul_transb(Tv,C2[2],tmpS);
      blas.matmul(D1[2],tmpS,Tw);

      tmpS2 -= Tw;

      Tv = tmpS2;
      Tv *= (tau_rk4/2);
      Tv += S;

      blas.matmul_transb(Tv,C1[0],tmpS);
      blas.matmul(D2[0],tmpS,tmpS3);
      blas.matmul_transb(Tv,C1[1],tmpS);
      blas.matmul(D2[1],tmpS,Tw);
      tmpS3 += Tw;
      blas.matmul_transb(Tv,C1[2],tmpS);
      blas.matmul(D2[2],tmpS,Tw);
      tmpS3 += Tw;
      blas.matmul_transb(Tv,C2[0],tmpS);
      blas.matmul(D1[0],tmpS,Tw);
      tmpS3 -= Tw;
      blas.matmul_transb(Tv,C2[1],tmpS);
      blas.matmul(D1[1],tmpS,Tw);
      tmpS3 -= Tw;
      blas.matmul_transb(Tv,C2[2],tmpS);
      blas.matmul(D1[2],tmpS,Tw);
      tmpS3 -= Tw;

      Tv = tmpS3;
      Tv *= tau_rk4;
      Tv += S;

      blas.matmul_transb(Tv,C1[0],tmpS);
      blas.matmul(D2[0],tmpS,tmpS4);
      blas.matmul_transb(Tv,C1[1],tmpS);
      blas.matmul(D2[1],tmpS,Tw);
      tmpS4 += Tw;
      blas.matmul_transb(Tv,C1[2],tmpS);
      blas.matmul(D2[2],tmpS,Tw);
      tmpS4 += Tw;
      blas.matmul_transb(Tv,C2[0],tmpS);
      blas.matmul(D1[0],tmpS,Tw);
      tmpS4 -= Tw;
      blas.matmul_transb(Tv,C2[1],tmpS);
      blas.matmul(D1[1],tmpS,Tw);
      tmpS4 -= Tw;
      blas.matmul_transb(Tv,C2[2],tmpS);
      blas.matmul(D1[2],tmpS,Tw);
      tmpS4 -= Tw;

      tmpS2 *= 2.0;
      tmpS3 *= 2.0;

      tmpS1 += tmpS2;
      tmpS1 += tmpS3;
      tmpS1 += tmpS4;
      tmpS1 *= (tau_rk4/6.0);

      S += tmpS1;
    }
  }
  
private:
  mat tmpS, tmpS1, tmpS2, tmpS3, tmpS4, Tv, Tw;
};


struct coeff_C {

  coeff_C(stloc sl, grid_info<3> _gi) : gi(_gi), tmpVhat(sl) {

    tmpVhat.resize({gi.dvvh_mult,gi.r});

    we       = create_vec_array(gi.dvv_mult, sl);
    dVhat    = create_cmat_array({gi.dvvh_mult,gi.r}, sl);
    dV       = create_mat_array({gi.dvv_mult,gi.r}, sl);
    h_lambda_n = create_cvec_array(gi.dvvh_mult, stloc::host);

    // Initialize we
    array<vec,3> h_we = create_vec_array(gi.dvv_mult, stloc::host);
    array<vec,3> v    = create_vec_array({gi.dvv_mult}, stloc::host);

    componentwise_vec_omp(gi.N_vv, [this, &v](Index idx, mind<3> i) {
      v[0](idx) = gi.v(0, i[0]);
      v[1](idx) = gi.v(1, i[1]);
      v[2](idx) = gi.v(2, i[2]);
    });

    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index j = 0; j < gi.dvv_mult; j++){
      h_we[0](j) = v[0](j) * gi.h_vv[0] * gi.h_vv[1] * gi.h_vv[2];
      h_we[1](j) = v[1](j) * gi.h_vv[0] * gi.h_vv[1] * gi.h_vv[2];
      h_we[2](j) = v[2](j) * gi.h_vv[0] * gi.h_vv[1] * gi.h_vv[2];
    }

    we[0] = h_we[0];
    we[1] = h_we[1];
    we[2] = h_we[2];

    double ncvv = 1.0 / (gi.dvv_mult);
    componentwise_vec_fourier_omp(gi.N_vv, [this, ncvv](Index idx, mind<3> i) {
      Index mult_j = freq(i[1], gi.N_vv[1]);
      Index mult_k = freq(i[2], gi.N_vv[2]);
      h_lambda_n[0](idx) = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[1]-gi.lim_vv[0])*i[0])*ncvv;
      h_lambda_n[1](idx) = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[3]-gi.lim_vv[2])*mult_j)*ncvv;
      h_lambda_n[2](idx) = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[5]-gi.lim_vv[4])*mult_k)*ncvv;
    });

      #ifdef __CUDACC__
      if(sl == stloc::device) {
        d_lim_vv = make_unique_ptr<vec>(array<Index,1>({6}), stloc::device);
        cudaMemcpy(d_lim_vv->data(), gi.lim_vv.data(), 6*sizeof(double), cudaMemcpyHostToDevice);
      }
      #endif
  }

  void operator()(mat& V, array<mat,3>& C1, array<mat,3>& C2, array<cmat,3>& C2c, const blas_ops& blas) {

      // C1
      coeff(V, V, we[0], C1[0], blas);
      coeff(V, V, we[1], C1[1], blas);
      coeff(V, V, we[2], C1[2], blas);

      // C2
      if(fft == nullptr)
        fft = make_unique_ptr<fft3d<2>>(gi.N_xx, V, tmpVhat);

      // TODO: relies on V not being overwritten
      fft->forward(V, tmpVhat);

      if(V.sl == stloc::host) {
        ptw_mult_row(tmpVhat,h_lambda_n[0],dVhat[0]);
        ptw_mult_row(tmpVhat,h_lambda_n[1],dVhat[1]);
        ptw_mult_row(tmpVhat,h_lambda_n[2],dVhat[2]);
      } else {
        #ifdef __CUDACC__
        double ncvv = 1.0 / (gi.dvv_mult);
        ptw_mult_row_cplx_fourier_3d<<<(gi.dvvh_mult*gi.r+n_threads-1)/n_threads,n_threads>>>(gi.dvvh_mult*gi.r, gi.N_vv[0]/2+1, gi.N_vv[1], gi.N_vv[2], (cuDoubleComplex*)tmpVhat.begin(), d_lim_vv->data(), ncvv, (cuDoubleComplex*)dVhat[0].begin(), (cuDoubleComplex*)dVhat[1].begin(), (cuDoubleComplex*)dVhat[2].begin());
        #endif
      }

      fft->backward(dVhat[0], dV[0]);
      fft->backward(dVhat[1], dV[1]);
      fft->backward(dVhat[2], dV[2]);

      coeff(V, dV[0], gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], C2[0], blas);
      coeff(V, dV[1], gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], C2[1], blas);
      coeff(V, dV[2], gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], C2[2], blas);

      if(V.sl == stloc::host) {
        C2[0].to_cplx(C2c[0]);
        C2[1].to_cplx(C2c[1]);
        C2[2].to_cplx(C2c[2]);
      } else {
        #ifdef __CUDACC__
        cplx_conv<<<(C2[0].num_elements()+n_threads-1)/n_threads,n_threads>>>(C2[0].num_elements(), C2[0].begin(), (cuDoubleComplex*)C2c[0].begin());
        cplx_conv<<<(C2[1].num_elements()+n_threads-1)/n_threads,n_threads>>>(C2[1].num_elements(), C2[1].begin(), (cuDoubleComplex*)C2c[1].begin());
        cplx_conv<<<(C2[2].num_elements()+n_threads-1)/n_threads,n_threads>>>(C2[2].num_elements(), C2[2].begin(), (cuDoubleComplex*)C2c[2].begin());
        #endif
      }
  }

private:
  grid_info<3> gi;
  array<vec,3> we;
  std::unique_ptr<fft3d<2>> fft;
  cmat tmpVhat;
  array<cmat,3> dVhat;
  array<mat,3> dV;
  array<cvec,3> h_lambda_n;
  std::unique_ptr<vec> d_lim_vv;
};

struct coeff_D {

  coeff_D(stloc sl, grid_info<3> _gi) : gi(_gi), tmpXhat(sl) {

    tmpXhat.resize({gi.dxxh_mult,gi.r});

    dX = create_mat_array({gi.dxx_mult,gi.r}, sl);
    we = create_vec_array(gi.dxx_mult, sl);
    dXhat = create_cmat_array({gi.dxxh_mult,gi.r}, sl);
    lambda_n = create_cvec_array(gi.dxxh_mult, sl);

    array<cvec,3> h_lambda_n = create_cvec_array(gi.dxxh_mult, stloc::host);

    double ncxx = 1.0 / (gi.dxx_mult);
    componentwise_vec_fourier_omp(gi.N_xx, [this, &h_lambda_n, ncxx](Index idx, mind<3> i) {
      Index mult_j = freq(i[1], gi.N_xx[1]);
      Index mult_k = freq(i[2], gi.N_xx[2]);
      h_lambda_n[0](idx) = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[1]-gi.lim_xx[0])*i[0])*ncxx;
      h_lambda_n[1](idx) = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[3]-gi.lim_xx[2])*mult_j)*ncxx;
      h_lambda_n[2](idx) = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[5]-gi.lim_xx[4])*mult_k)*ncxx;
    });
    
    lambda_n[0] = h_lambda_n[0];
    lambda_n[1] = h_lambda_n[1];
    lambda_n[2] = h_lambda_n[2];
  }

  void operator()(mat& X, array<vec, 3>& E, array<mat,3>& D1, array<mat,3>& D2, const blas_ops& blas) {

      if(X.sl == stloc::host) {
        #ifdef __OPENMP__
        #pragma omp parallel for
        #endif
        for(Index j = 0; j < gi.dxx_mult; j++){
          we[0](j) = E[0](j) * gi.h_xx[0] * gi.h_xx[1] * gi.h_xx[2];
          we[1](j) = E[1](j) * gi.h_xx[0] * gi.h_xx[1] * gi.h_xx[2];
          we[2](j) = E[2](j) * gi.h_xx[0] * gi.h_xx[1] * gi.h_xx[2];
        }
      } else {
        #ifdef __CUDACC__
        ptw_mult_scal<<<(E[0].num_elements()+n_threads-1)/n_threads,n_threads>>>(E[0].num_elements(), E[0].begin(), gi.h_xx[0] * gi.h_xx[1] * gi.h_xx[2], we[0].begin());
        ptw_mult_scal<<<(E[1].num_elements()+n_threads-1)/n_threads,n_threads>>>(E[1].num_elements(), E[1].begin(), gi.h_xx[0] * gi.h_xx[1] * gi.h_xx[2], we[1].begin());
        ptw_mult_scal<<<(E[2].num_elements()+n_threads-1)/n_threads,n_threads>>>(E[2].num_elements(), E[2].begin(), gi.h_xx[0] * gi.h_xx[1] * gi.h_xx[2], we[2].begin());
        #endif
      }

      coeff(X, X, we[0], D1[0], blas);
      coeff(X, X, we[1], D1[1], blas);
      coeff(X, X, we[2], D1[2], blas);

      if(fft == nullptr)
        fft = make_unique_ptr<fft3d<2>>(gi.N_xx, X, tmpXhat);

      fft->forward(X, tmpXhat);

      ptw_mult_row(tmpXhat,lambda_n[0],dXhat[0]);
      ptw_mult_row(tmpXhat,lambda_n[1],dXhat[1]);
      ptw_mult_row(tmpXhat,lambda_n[2],dXhat[2]);
      /* TODO: direct multiplicaiton on the GPU is probably faster
      ptw_mult_row_cplx_fourier_3d<<<(dxxh_mult*r+n_threads-1)/n_threads,n_threads>>>(dxxh_mult*r, N_xx[0]/2+1, N_xx[1], N_xx[2], d_tmpXhat.begin(), d_lim_xx, ncxx, d_dXhat_x.begin(), d_dXhat_y.begin(), d_dXhat_z.begin());
      */

      fft->backward(dXhat[0], dX[0]);
      fft->backward(dXhat[1], dX[1]);
      fft->backward(dXhat[2], dX[2]);

      coeff(X, dX[0], gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2], D2[0], blas);
      coeff(X, dX[1], gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2], D2[1], blas);
      coeff(X, dX[2], gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2], D2[2], blas);
  }

private:
  grid_info<3> gi;
  array<vec,3> we;
  std::unique_ptr<fft3d<2>> fft;
  cmat tmpXhat;
  array<cmat,3> dXhat;
  array<cvec,3> lambda_n;
  array<mat,3>  dX;
};


struct electric_field {

  electric_field(stloc sl, grid_info<3> _gi)
    : gi(_gi), int_V(sl), ef(sl), efhat(sl) {

    ef.resize({gi.dxx_mult});

    int_V.resize({gi.r});

    efhat.resize({gi.dxxh_mult});
    efhatx = create_cvec_array({gi.dxxh_mult}, sl);

    #ifdef __CUDACC__
    if(sl == stloc::device) {
      d_lim_xx = make_unique_ptr<vec>(array<Index,1>({6}), stloc::device);
      cudaMemcpy(d_lim_xx->data(), gi.lim_xx.data(), 6*sizeof(double), cudaMemcpyHostToDevice);
    }
    #endif
  }

  void operator()(const mat& K, const mat& V, array<vec,3>& E, const blas_ops& blas) {

      integrate(V,-gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2],int_V,blas);
      blas.matvec(K,int_V,ef);
      ef += 1.0;

      if(fft == nullptr)
        fft = make_unique_ptr<fft3d<1>>(gi.N_xx, ef, efhat);

      fft->forward(ef, efhat);

      double ncxx = 1.0 / double(gi.dxx_mult);
      if(K.sl == stloc::host) {
        componentwise_vec_fourier_omp(gi.N_xx, [this, ncxx](Index idx, mind<3> i) {
          Index mult_j = freq(i[1], gi.N_xx[1]);
          Index mult_k = freq(i[2], gi.N_xx[2]);
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[1]-gi.lim_xx[0])*i[0]);
          complex<double> lambday = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[3]-gi.lim_xx[2])*mult_j);
          complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[5]-gi.lim_xx[4])*mult_k);
              
          efhatx[0](idx) = efhat(idx) * lambdax / (pow(lambdax,2) + pow(lambday,2) + pow(lambdaz,2)) * ncxx;
          efhatx[1](idx) = efhat(idx) * lambday / (pow(lambdax,2) + pow(lambday,2) + pow(lambdaz,2)) * ncxx ;
          efhatx[2](idx) = efhat(idx) * lambdaz / (pow(lambdax,2) + pow(lambday,2) + pow(lambdaz,2)) * ncxx ;
        });

        // TODO: this can be simplified.
        #ifdef __OPENMP__
        #pragma omp parallel for
        #endif
        for(Index k = 0; k < (gi.N_xx[2]/2 + 1); k += (gi.N_xx[2]/2)){
          for(Index j = 0; j < (gi.N_xx[1]/2 + 1); j += (gi.N_xx[1]/2)){
            efhatx[0](j*(gi.N_xx[0]/2+1) + k*((gi.N_xx[0]/2+1)*gi.N_xx[1])) = complex<double>(0.0,0.0);
            efhatx[1](j*(gi.N_xx[0]/2+1) + k*((gi.N_xx[0]/2+1)*gi.N_xx[1])) = complex<double>(0.0,0.0);
            efhatx[2](j*(gi.N_xx[0]/2+1) + k*((gi.N_xx[0]/2+1)*gi.N_xx[1])) = complex<double>(0.0,0.0);
          }
        }
      } else {
        #ifdef __CUDACC__
        der_fourier_3d<<<(gi.dxxh_mult+n_threads-1)/n_threads,n_threads>>>(gi.dxxh_mult, gi.N_xx[0]/2+1, gi.N_xx[1], gi.N_xx[2], (cuDoubleComplex*)efhat.begin(), d_lim_xx->data(), ncxx, (cuDoubleComplex*)efhatx[0].begin(), (cuDoubleComplex*)efhatx[1].begin(), (cuDoubleComplex*)efhatx[2].begin());
        #endif
      }

      fft->backward(efhatx[0], E[0]);
      fft->backward(efhatx[1], E[1]);
      fft->backward(efhatx[2], E[2]);
  }

private:
  grid_info<3> gi;
  std::unique_ptr<fft3d<1>> fft;
  vec int_V, ef;
  cvec efhat;
  array<cvec,3> efhatx;
  std::unique_ptr<vec> d_lim_xx;
};


void save_lr(string fn, const lr2<double>& lr_sol, const grid_info<3>& gi) {
    nc_writer ncw(fn, {gi.N_xx[0], gi.N_xx[1], gi.N_xx[2], gi.N_vv[0], gi.N_vv[1], gi.N_vv[2], gi.r}, {"x", "y", "z", "v", "w", "u", "r"});
    ncw.add_var("r", {"r"});
    ncw.add_var("x", {"x"});
    ncw.add_var("y", {"y"});
    ncw.add_var("z", {"z"});
    ncw.add_var("u", {"u"});
    ncw.add_var("v", {"v"});
    ncw.add_var("w", {"w"});
    ncw.add_var("X", {"r", "z", "y", "x"});
    ncw.add_var("S", {"r", "r"});
    ncw.add_var("V", {"r", "u", "w", "v"});

    ncw.start_write_mode();

    vector<double> vec_r(gi.r);
    for(Index i=0;i<gi.r;i++)
      vec_r[i] = i;

    vector<double> vec_x(gi.N_xx[0]), vec_y(gi.N_xx[1]), vec_z(gi.N_xx[2]);
    for(Index i=0;i<gi.N_xx[0];i++)
        vec_x[i] = gi.x(0, i);
    for(Index i=0;i<gi.N_xx[1];i++)
        vec_y[i] = gi.x(1, i);
    for(Index i=0;i<gi.N_xx[2];i++)
        vec_z[i] = gi.x(2, i);

    vector<double> vec_v(gi.N_vv[0]), vec_w(gi.N_vv[1]), vec_u(gi.N_vv[2]);
    for(Index i=0;i<gi.N_vv[0];i++)
        vec_v[i] = gi.v(0, i);
    for(Index i=0;i<gi.N_vv[1];i++)
        vec_w[i] = gi.v(1, i);
    for(Index i=0;i<gi.N_vv[2];i++)
        vec_u[i] = gi.v(2, i);

    ncw.write("r", vec_r.data());
    ncw.write("x", vec_x.data());
    ncw.write("y", vec_y.data());
    ncw.write("z", vec_z.data());
    ncw.write("v", vec_v.data());
    ncw.write("w", vec_w.data());
    ncw.write("u", vec_u.data());

    ncw.write("X", lr_sol.X.data());
    ncw.write("S", lr_sol.S.data());
    ncw.write("V", lr_sol.V.data());
}

void integration_first_order(double final_time, double tau, int nsteps_split, int nsteps_ei, int nsteps_rk4, const grid_info<3>& gi, vector<const double*> X0, vector<const double*> V0, Index snapshots, const blas_ops& blas){

  stloc sl = (CPU) ? stloc::host : stloc::device;

  orthogonalize gs(&blas);

  // For C coefficients
  coeff_C compute_C(sl, gi);
  coeff_D compute_D(sl, gi);
  electric_field efield(sl, gi);

  array<mat, 3> C1   = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> C2   = create_mat_array({gi.r,gi.r}, sl);
  array<cmat, 3> C2c = create_cmat_array({gi.r,gi.r}, sl);
  
  array<mat, 3> D1   = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> D2   = create_mat_array({gi.r,gi.r}, sl);

  PS_K_step K_step(sl, gi, &blas);
  PS_S_step S_step(sl, gi.r);
  PS_L_step L_step(sl, gi, &blas);

  mat tmpX({gi.dxx_mult,gi.r}, sl);
  mat tmpV({gi.dvv_mult,gi.r}, sl);

  // Electric field
  array<vec,3> E = create_vec_array(gi.dxx_mult, sl);

  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2], gi.dxx_mult);
  std::function<double(double*,double*)> ip_vv = inner_product_from_const_weight(gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], gi.dvv_mult);

  // initialization
  lr2<double> lr_sol(gi.r,{gi.dxx_mult,gi.dvv_mult}, sl);
  if(sl == stloc::host) {
    initialize(lr_sol, X0, V0, ip_xx, ip_vv, blas);
  } else {
    lr2<double> h_lr_sol(gi.r,{gi.dxx_mult,gi.dvv_mult}, stloc::host);
    initialize(h_lr_sol, X0, V0, ip_xx, ip_vv, blas);
    lr_sol = h_lr_sol;
  }

  ofstream el_energyf("evolution.data");
  double t = 0.0;
  Index n_steps = ceil(final_time/tau);
  for(Index ts=0;ts<n_steps;ts++) {
    if(final_time - t < tau)
      tau = final_time - t;
/* TODO
    if(snapshots>=2 && (ts % int(ceil(n_steps/double(snapshots-1))) == 0)) {
      std::stringstream ss;
      ss << "lr-t" << t << ".nc";
      save_lr(ss.str(), lr_sol, gi);
    }
*/
    // Compute K and store it in lr_sol.X
    tmpX = lr_sol.X;
    blas.matmul(tmpX,lr_sol.S,lr_sol.X);

    // compute electric field
    efield(lr_sol.X, lr_sol.V, E, blas);

    // K step
    compute_C(lr_sol.V, C1, C2, C2c, blas);

    K_step(tau, lr_sol.X, E, C1, C2c, nsteps_ei);
    
    if(lr_sol.X.sl == stloc::host)
      gs(lr_sol.X, lr_sol.S, ip_xx);
    else
      // TODO: only support for constant inner product on the GPU for now
      gs(lr_sol.X, lr_sol.S, gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);

    // S step
    compute_D(lr_sol.X, E, D1, D2, blas);

    S_step(tau, lr_sol.S, C1, C2, D1, D2, blas);
    
    // L step
    tmpV = lr_sol.V;
    blas.matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    L_step(tau, lr_sol.V, D1, D2);

    // TODO
    if(lr_sol.V.sl == stloc::host)
      gs(lr_sol.V, lr_sol.S, ip_vv);
    else
      gs(lr_sol.V, lr_sol.S, gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);


    transpose_inplace(lr_sol.S);

    double el_energy = 0.0;
    if(E[0].sl == stloc::host) {
      #ifdef __OPENMP__
      #pragma omp parallel for reduction(+:el_energy)
      #endif
      for(Index ii = 0; ii < gi.dxx_mult; ii++){
        el_energy += 0.5*(pow(E[0](ii),2)+pow(E[1](ii),2)+pow(E[2](ii),2))*gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2];
      }
    } else {
      // TODO: can be improved and simplified
      #ifdef __CUDACC__
      double* d_el_energy;
      cudaMalloc(&d_el_energy, sizeof(double)*3);
      cublasDdot (blas.handle_devres, E[0].num_elements(), E[0].begin(), 1, E[0].begin(), 1, d_el_energy);
      cublasDdot (blas.handle_devres, E[1].num_elements(), E[1].begin(), 1, E[1].begin(), 1, d_el_energy+1);
      cublasDdot (blas.handle_devres, E[2].num_elements(), E[2].begin(), 1, E[2].begin(), 1, d_el_energy+2);
      cudaDeviceSynchronize();
      ptw_sum<<<1,1>>>(1,d_el_energy,d_el_energy+1);
      ptw_sum<<<1,1>>>(1,d_el_energy,d_el_energy+2);

      scale_unique<<<1,1>>>(d_el_energy,0.5*gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);

      cudaMemcpy(&el_energy,d_el_energy,sizeof(double),cudaMemcpyDeviceToHost);
      cudaFree(d_el_energy);
      #endif
    }

    el_energyf << t << " " << el_energy << endl;

    t += tau;
  }
/* TODO 
  if(snapshots>=1) {
    std::stringstream ss;
    ss << "lr-t" << t << ".nc";
    save_lr(ss.str(), lr_sol, gi);
  }*/
}


int main(int argc, char** argv){

  cxxopts::Options options("vlasov_poisson", "3+3 dimensional dynamical low-rank Vlasov--Poisson solver");
  options.add_options()
  ("device", "Device the simulation is run on (can be either cpu or gpu)", cxxopts::value<string>()->default_value("cpu"))
  ("problem", "Initial value that is used in the simulation (either ll or ts)", cxxopts::value<string>()->default_value("ts"))
  ("final_time", "Time to which the simulation is run", cxxopts::value<double>()->default_value("60.0"))
  ("deltat", "The time step used in the simulation (usually denoted by \\Delta t or tau)", cxxopts::value<double>()->default_value("0.01"))
  ("r,rank", "Rank of the simulation", cxxopts::value<int>()->default_value("10"))
  ("nx", "Number of grid points in space (as a whitespace separated list)", cxxopts::value<string>()->default_value("16 16 16"))
  ("nv", "Number of grid points in velocity (as a whitespace separated list)", cxxopts::value<string>()->default_value("16 16 16"))
  ("omp_threads", "Number of OpenMP threads used in CPU parallelization (by default half the number of processes reported by the operating system are used)", cxxopts::value<int>()->default_value("-1"))
  ("snapshots", "Number of files written to disk", cxxopts::value<int>()->default_value("0"))
  ("h,help", "Help message")
  ;
  auto result = options.parse(argc, argv);

  if(result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  #ifndef __CUDACC__
  CPU = true;
  #else
  string dev = result["device"].as<string>();
  if(dev == "cpu")
    CPU = true;
  else if(dev == "gpu")
    CPU = false;
  else {
    cout << "ERROR: device " << dev << " not found." << endl;
    exit(1);
  }
  #endif

  array<Index,3> N_xx = parse<3>(result["nx"].as<string>());
  array<Index,3> N_vv = parse<3>(result["nv"].as<string>());

  #ifdef __OPENMP__
  int num_threads = result["omp_threads"].as<int>();
  if(num_threads == -1)
    num_threads = omp_get_num_procs()/2;
  omp_set_num_threads(num_threads);

  #pragma omp parallel
  {
    if(omp_get_thread_num()==0){
      cout << "Number of threads: " << omp_get_num_threads() << endl;
    }
  }
  #endif

  Index   r = result["rank"].as<int>();
  double  final_time = result["final_time"].as<double>();
  double  tau = result["deltat"].as<double>();
  Index snapshots = result["snapshots"].as<int>();

  int nsteps_split = 1;
  int nsteps_ei = 1;
  int nsteps_rk4 = 1;

  blas_ops blas(!CPU);

  // Setup the initial value
  string problem = result["problem"].as<string>();
  if(problem == "ll") {
    //
    // Landau damping
    //
    mfp<6> lim_xx = {0.0,4.0*M_PI,0.0,4.0*M_PI,0.0,4.0*M_PI}; // Limits for box [ax,bx] x [ay,by] x [az,bz] {ax,bx,ay,by,az,bz}
    mfp<6> lim_vv = {-6.0,6.0,-6.0,6.0,-6.0,6.0}; // Limits for box [av,bv] x [aw,bw] x [au,bu] {av,bv,aw,bw,au,bu}
    grid_info<3> gi(r, N_xx, N_vv, lim_xx, lim_vv); 

    

    vec xx({gi.dxx_mult});
    componentwise_vec_omp(gi.N_xx, [&xx, &gi](Index idx, array<Index,3> i) {
      double alpha1 = 0.01, alpha2 = 0.01, alpha3 = 0.01;
      double kappa1 = 0.5, kappa2 = 0.5, kappa3 = 0.5;
      mfp<3> x  = gi.x(i);
      xx(idx) = 1.0 + alpha1*cos(kappa1*x[0]) + alpha2*cos(kappa2*x[1]) + alpha3*cos(kappa3*x[2]);
    });

    vec vv({gi.dvv_mult});
    componentwise_vec_omp(gi.N_vv, [&vv, &gi](Index idx, array<Index,3> i) {
        mfp<3> v  = gi.v(i);
        vv(idx) = (1.0/(sqrt(pow(2*M_PI,3)))) * exp(-(pow(v[0],2)+pow(v[1],2)+pow(v[2],2))/2.0);
    });

    vector<const double*> X, V;
    X.push_back(xx.begin());
    V.push_back(vv.begin());

    integration_first_order(final_time, tau, nsteps_split, nsteps_ei, nsteps_rk4, gi, X, V, snapshots, blas);
  } else if(problem == "ts") {
    //
    // Two-stream instability
    //
    mfp<6> lim_xx = {0.0,10.0*M_PI,0.0,10.0*M_PI,0.0,10.0*M_PI};
    mfp<6> lim_vv = {-9.0,9.0,-9.0,9.0,-9.0,9.0};
    grid_info<3> gi(r, N_xx, N_vv, lim_xx, lim_vv); 

    vec xx({gi.dxx_mult});
    componentwise_vec_omp(gi.N_xx, [&xx, &gi](Index idx, array<Index,3> i) {
      double alpha1 = 0.001, alpha2 = 0.001, alpha3 = 0.001;
      double kappa1 = 1.0/5.0, kappa2 = 1.0/5.0, kappa3=1.0/5.0;
      mfp<3> x  = gi.x(i);
      xx(idx) = 1.0 + alpha1*cos(kappa1*x[0]) + alpha2*cos(kappa2*x[1]) + alpha3*cos(kappa3*x[2]);
    });

    vec vv({gi.dvv_mult});
    componentwise_vec_omp(gi.N_vv, [&vv, &gi](Index idx, array<Index,3> i) {
        double v0 = 2.5, w0 = 0.0, u0=0.0;
        double v0b = -2.5, w0b = -2.25, u0b = -2.0;
        mfp<3> v  = gi.v(i);
        vv(idx) = (1.0/(sqrt(pow(8*M_PI,3)))) * (exp(-(pow(v[0]-v0,2))/2.0)+exp(-(pow(v[0]-v0b,2))/2.0))*(exp(-(pow(v[1]-w0,2))/2.0)+exp(-(pow(v[1]-w0b,2))/2.0))*(exp(-(pow(v[2]-u0,2))/2.0)+exp(-(pow(v[2]-u0b,2))/2.0));
    });

    vector<const double*> X, V;
    X.push_back(xx.begin());
    V.push_back(vv.begin());

    integration_first_order(final_time, tau, nsteps_split, nsteps_ei, nsteps_rk4, gi, X, V, snapshots, blas);
  } else {
    cout << "ERROR: problem with name " << problem << " is not supported" << endl;
    exit(1);
  }

  return 0;
}
