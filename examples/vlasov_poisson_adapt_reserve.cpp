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

using namespace Ensign;
using namespace Ensign::Matrix;

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
struct grid_info_reserve {
  Index r;
  Index rmax;
  mind<d>  N_xx, N_vv;
  mfp<2*d> lim_xx, lim_vv;
  mfp<d>   h_xx, h_vv;
  Index dxx_mult, dvv_mult, dxxh_mult, dvvh_mult;

  grid_info_reserve(Index _rmax, Index _r, mind<d> _N_xx, mind<d> _N_vv):
   r(_r), rmax(_rmax), N_xx(_N_xx), N_vv(_N_vv) {
    dxx_mult  = N_xx[0]*N_xx[1]*N_xx[2];
    dvv_mult  = N_vv[0]*N_vv[1]*N_vv[2];
    dxxh_mult = N_xx[2]*N_xx[1]*(N_xx[0]/2 + 1);
    dvvh_mult = N_vv[2]*N_vv[1]*(N_vv[0]/2 + 1);
   }

   void operator()(mfp<2*d> _lim_xx, mfp<2*d> _lim_vv) {
     lim_xx = _lim_xx;
     lim_vv = _lim_vv;
     // compute h_xx and h_vv
     for(int ii = 0; ii < 3; ii++){
       Index jj = 2*ii;
       h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
       h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
     }
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

  void update_rank(Index _r) {
    r = _r;
  }
};

array<mat,4> create_rk4_array(mind<2> dim, stloc sl) {
  return {mat(dim,sl), mat(dim,sl), mat(dim,sl), mat(dim,sl)};
}

array<mat,4> initialize_rk4_array(stloc sl) {
  return {mat({1,1},sl), mat({1,1},sl), mat({1,1},sl), mat({1,1},sl)};
}

array<mat,3> create_mat_array(mind<2> dim, stloc sl) {
  return {mat(dim,sl), mat(dim,sl), mat(dim,sl)};
}

array<mat,3> initialize_mat_array(stloc sl) {
  return {mat({1,1},sl), mat({1,1},sl), mat({1,1},sl)};
}

array<cmat,3> initialize_cmat_array(stloc sl) {
  return {cmat({1,1},sl), cmat({1,1},sl), cmat({1,1},sl)};
}

array<vec,3> create_vec_array(Index dim, stloc sl) {
  return {vec({dim}, sl), vec({dim}, sl), vec({dim}, sl)};
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

struct coeff_C {

  coeff_C(stloc sl, grid_info_reserve<3> _gi) : gi(_gi), tmpVhat(sl) {

    tmpVhat.reserve({gi.dvvh_mult,gi.rmax},{gi.dvvh_mult,gi.r});

    we       = create_vec_array(gi.dvv_mult, sl);
    dVhat = initialize_cmat_array(sl);
    dV = initialize_mat_array(sl);
    for(int i = 0; i < 3; i++){
      dVhat[i].reserve({gi.dvvh_mult,gi.rmax},{gi.dvvh_mult,gi.r});
      dV[i].reserve({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r});
    }
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

      #ifdef __CUDA__
      if(sl == stloc::device) {
        d_lim_vv = make_unique_ptr<vec>(array<Index,1>({6}), stloc::device);
        cudaMemcpy(d_lim_vv->data(), gi.lim_vv.data(), 6*sizeof(double), cudaMemcpyHostToDevice);
      }
      #endif
  }
  
  void operator()(mat& V, array<mat,3>& C1, array<mat,3>& C2, const blas_ops& blas) {

      // C1
      coeff(V, V, we[0], C1[0], blas);
      coeff(V, V, we[1], C1[1], blas);
      coeff(V, V, we[2], C1[2], blas);

      // C2
      if(fft == nullptr)
        fft = make_unique_ptr<fft3d<2>>(gi.N_vv, V, tmpVhat, true);

      for(Index kk = 0; kk < gi.r; kk ++){
        fft->forward(V.begin()+kk*gi.dvv_mult, tmpVhat.begin()+kk*gi.dvvh_mult,V.sl);
      }

      if(V.sl == stloc::host) {
        ptw_mult_row(tmpVhat,h_lambda_n[0],dVhat[0]);
        ptw_mult_row(tmpVhat,h_lambda_n[1],dVhat[1]);
        ptw_mult_row(tmpVhat,h_lambda_n[2],dVhat[2]);
      } else {
        #ifdef __CUDA__
        double ncvv = 1.0 / (gi.dvv_mult);
        ptw_mult_row_cplx_fourier_3d<<<(gi.dvvh_mult*gi.r+n_threads-1)/n_threads,n_threads>>>(gi.dvvh_mult*gi.r, gi.N_vv[0]/2+1, gi.N_vv[1], gi.N_vv[2], (cuDoubleComplex*)tmpVhat.begin(), d_lim_vv->data(), ncvv, (cuDoubleComplex*)dVhat[0].begin(), (cuDoubleComplex*)dVhat[1].begin(), (cuDoubleComplex*)dVhat[2].begin());
        #endif
      }

      for(Index kk = 0; kk < gi.r; kk ++){
        fft->backward(dVhat[0].begin()+kk*gi.dvvh_mult, dV[0].begin()+kk*gi.dvv_mult,dV[0].sl);
        fft->backward(dVhat[1].begin()+kk*gi.dvvh_mult, dV[1].begin()+kk*gi.dvv_mult,dV[1].sl);
        fft->backward(dVhat[2].begin()+kk*gi.dvvh_mult, dV[2].begin()+kk*gi.dvv_mult,dV[2].sl);
      }


      coeff(V, dV[0], gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], C2[0], blas);
      coeff(V, dV[1], gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], C2[1], blas);
      coeff(V, dV[2], gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], C2[2], blas);

  }

  void update_info(Index nr){
    gi.update_rank(nr);
    tmpVhat.update_shape({gi.dvvh_mult,gi.r});
    for(int i=0; i<3; i++){
      dVhat[i].update_shape({gi.dvvh_mult,gi.r});
      dV[i].update_shape({gi.dvv_mult,gi.r});
    }

  }

private:
  grid_info_reserve<3> gi;
  array<vec,3> we;
  std::unique_ptr<fft3d<2>> fft;
  cmat tmpVhat;
  array<cmat,3> dVhat;
  array<mat,3> dV;
  array<cvec,3> h_lambda_n;
  std::unique_ptr<vec> d_lim_vv;
};

struct coeff_D {

  coeff_D(stloc sl, grid_info_reserve<3> _gi) : gi(_gi), tmpXhat(sl) {

    tmpXhat.reserve({gi.dxxh_mult,gi.rmax},{gi.dxxh_mult,gi.r});

    dXhat = initialize_cmat_array(sl);
    dX = initialize_mat_array(sl);
    for(int i = 0; i < 3; i++){
      dXhat[i].reserve({gi.dxxh_mult,gi.rmax},{gi.dxxh_mult,gi.r});
      dX[i].reserve({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r});
    }

    we = create_vec_array(gi.dxx_mult, sl);
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
        #ifdef __CUDA__
        ptw_mult_scal<<<(E[0].num_elements()+n_threads-1)/n_threads,n_threads>>>(E[0].num_elements(), E[0].begin(), gi.h_xx[0] * gi.h_xx[1] * gi.h_xx[2], we[0].begin());
        cudaDeviceSynchronize();
        ptw_mult_scal<<<(E[1].num_elements()+n_threads-1)/n_threads,n_threads>>>(E[1].num_elements(), E[1].begin(), gi.h_xx[0] * gi.h_xx[1] * gi.h_xx[2], we[1].begin());
        cudaDeviceSynchronize();
        ptw_mult_scal<<<(E[2].num_elements()+n_threads-1)/n_threads,n_threads>>>(E[2].num_elements(), E[2].begin(), gi.h_xx[0] * gi.h_xx[1] * gi.h_xx[2], we[2].begin());
        cudaDeviceSynchronize();
        #endif
      }

      coeff(X, X, we[0], D1[0], blas);
      coeff(X, X, we[1], D1[1], blas);
      coeff(X, X, we[2], D1[2], blas);

      if(fft == nullptr)
        fft = make_unique_ptr<fft3d<2>>(gi.N_xx, X, tmpXhat,true);

      for(Index kk = 0; kk < gi.r; kk ++){
        fft->forward(X.begin()+kk*gi.dxx_mult, tmpXhat.begin()+kk*gi.dxxh_mult,X.sl);
      }

      ptw_mult_row(tmpXhat,lambda_n[0],dXhat[0]);
      ptw_mult_row(tmpXhat,lambda_n[1],dXhat[1]);
      ptw_mult_row(tmpXhat,lambda_n[2],dXhat[2]);

      for(Index kk = 0; kk < gi.r; kk ++){
        fft->backward(dXhat[0].begin()+kk*gi.dxxh_mult, dX[0].begin()+kk*gi.dxx_mult,dX[0].sl);
        fft->backward(dXhat[1].begin()+kk*gi.dxxh_mult, dX[1].begin()+kk*gi.dxx_mult,dX[1].sl);
        fft->backward(dXhat[2].begin()+kk*gi.dxxh_mult, dX[2].begin()+kk*gi.dxx_mult,dX[2].sl);
      }

      coeff(X, dX[0], gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2], D2[0], blas);
      coeff(X, dX[1], gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2], D2[1], blas);
      coeff(X, dX[2], gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2], D2[2], blas);
  }
  
  void update_info(Index nr){
    gi.update_rank(nr);
    tmpXhat.update_shape({gi.dxxh_mult,gi.r});
    for(int i=0; i<3; i++){
      dXhat[i].update_shape({gi.dxxh_mult,gi.r});
      dX[i].update_shape({gi.dxx_mult,gi.r});
    }

  }

private:
  grid_info_reserve<3> gi;
  array<vec,3> we;
  std::unique_ptr<fft3d<2>> fft;
  cmat tmpXhat;
  array<cmat,3> dXhat;
  array<cvec,3> lambda_n;
  array<mat,3>  dX;
};


struct electric_field {

  electric_field(stloc sl, grid_info_reserve<3> _gi)
    : gi(_gi), int_V(sl), ef(sl), efhat(sl) {

    ef.resize({gi.dxx_mult});

    int_V.resize({gi.r});

    efhat.resize({gi.dxxh_mult});
    efhatx = create_cvec_array({gi.dxxh_mult}, sl);

    #ifdef __CUDA__
    if(sl == stloc::device) {
      d_lim_xx = make_unique_ptr<vec>(array<Index,1>({6}), stloc::device);
      cudaMemcpy(d_lim_xx->data(), gi.lim_xx.data(), 6*sizeof(double), cudaMemcpyHostToDevice);
    }
    #endif
  }

  void update_info(Index nr){
    gi.update_rank(nr);
    int_V.resize({gi.r});
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
        #ifdef __CUDA__
        der_fourier_3d<<<(gi.dxxh_mult+n_threads-1)/n_threads,n_threads>>>(gi.dxxh_mult, gi.N_xx[0]/2+1, gi.N_xx[1], gi.N_xx[2], (cuDoubleComplex*)efhat.begin(), d_lim_xx->data(), ncxx, (cuDoubleComplex*)efhatx[0].begin(), (cuDoubleComplex*)efhatx[1].begin(), (cuDoubleComplex*)efhatx[2].begin());
        #endif
      }

      fft->backward(efhatx[0], E[0]);
      fft->backward(efhatx[1], E[1]);
      fft->backward(efhatx[2], E[2]);
  }

private:
  grid_info_reserve<3> gi;
  std::unique_ptr<fft3d<1>> fft;
  vec int_V, ef;
  cvec efhat;
  array<cvec,3> efhatx;
  std::unique_ptr<vec> d_lim_xx;
};


void save_lr(string fn, const lr2<double>& lr_sol, const grid_info_reserve<3>& gi) {
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

void addsub_rhs(mat& A, mat& B, mat& C){
  if(A.sl == stloc::host){
    Index n = A.shape()[0];
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index i = 0; i < A.num_elements(); i++){
        Index r = i%n;
        Index c = i/n;
        A(r,c) += (B(r,c) - C(r,c));
    }
  } else {
    #ifdef __CUDA__
      addsub_rhs_k<<<(A.num_elements()+n_threads-1)/n_threads,n_threads>>>(A.num_elements(),A.begin(),B.begin(),C.begin());
    #endif
  }
}

void setmultadd_rk4(mat& A, mat& B, double alpha, mat& C){
  if(A.sl == stloc::host){
    Index n = A.shape()[0];
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index i = 0; i < A.num_elements(); i++){
        Index r = i%n;
        Index c = i/n;
        A(r,c) = B(r,c) + alpha*C(r,c);
    }
  } else {
    #ifdef __CUDA__
      setmultadd_rk4_k<<<(A.num_elements()+n_threads-1)/n_threads,n_threads>>>(A.num_elements(),A.begin(),B.begin(),alpha,C.begin());
    #endif
  }
}

void finstage_rk4(mat& A, mat& B, mat& C, mat& D, mat& E, double tau){
  if(A.sl == stloc::host){
    Index n = A.shape()[0];
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index i = 0; i < A.num_elements(); i++){
        Index r = i%n;
        Index c = i/n;
        A(r,c) = A(r,c) + (tau/6.0)*(B(r,c)+2.0*(C(r,c)+D(r,c))+E(r,c));
    }
  } else {
    #ifdef __CUDA__
      finstage_rk4_k<<<(A.num_elements()+n_threads-1)/n_threads,n_threads>>>(A.num_elements(),A.begin(),B.begin(),C.begin(),D.begin(),E.begin(),tau);
    #endif
  }
}

struct PS_K_step_adapt {

  PS_K_step_adapt(stloc _sl, grid_info_reserve<3> _gi, const blas_ops* _blas)
    : sl(_sl), gi(_gi), blas(_blas), fft(nullptr), Uhat(_sl), tmpX(_sl), tmpX2(_sl){

      Uhat.reserve({gi.dxxh_mult,gi.rmax},{gi.dxxh_mult,gi.r});
    
      UUhat = initialize_cmat_array(sl);
      UU = initialize_mat_array(sl);
      KK = initialize_rk4_array(sl);
      for(int i = 0; i < 3; i++){
        UUhat[i].reserve({gi.dxxh_mult,gi.rmax},{gi.dxxh_mult,gi.r});
        UU[i].reserve({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r});
        KK[i].reserve({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r});
      }
      KK[3].reserve({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r});

      tmpX.reserve({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r});
      tmpX2.reserve({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r});
    
      #ifdef __CUDA__
      if(sl == stloc::device) {
        d_lim_xx = make_unique_ptr<vec>(array<Index,1>({6}), stloc::device);
        cudaMemcpy(d_lim_xx->data(), gi.lim_xx.data(), 6*sizeof(double), cudaMemcpyHostToDevice);
      }
      #endif

    }


  void rk4_K_rhs(mat& U, array<vec,3>& ef, const array<mat,3>& C1, const array<mat,3>& C2, mat& out){

    // Perform needed derivatives in Fourier space
    // TODO: if needed we could work on a single column
    for(Index kk = 0; kk < gi.r; kk ++){
      fft->forward(U.begin()+kk*gi.dxx_mult,Uhat.begin()+kk*gi.dxxh_mult,U.sl);
    }
    double ncxx = 1.0 / double(gi.dxx_mult);
    if(sl == stloc::host){
      componentwise_mat_fourier_omp(gi.r, gi.N_xx, [this, ncxx](Index idx, mind<3> i, Index rr) {
      Index mult_j = freq(i[1], gi.N_xx[1]);
      Index mult_k = freq(i[2], gi.N_xx[2]);
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[1]-gi.lim_xx[0])*i[0]);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[3]-gi.lim_xx[2])*mult_j);
      complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[5]-gi.lim_xx[4])*mult_k);

      UUhat[0](idx,rr) = Uhat(idx,rr) * lambdax * ncxx;
      UUhat[1](idx,rr) = Uhat(idx,rr) * lambday * ncxx ;
      UUhat[2](idx,rr) = Uhat(idx,rr) * lambdaz * ncxx ;
      });
    } else {
      #ifdef __CUDA__
        ptw_mult_row_cplx_fourier_3d<<<(gi.dxxh_mult*gi.r+n_threads-1)/n_threads,n_threads>>>(gi.dxxh_mult*gi.r, gi.N_xx[0]/2+1, gi.N_xx[1], gi.N_xx[2], (cuDoubleComplex*)Uhat.begin(), d_lim_xx->data(), ncxx, (cuDoubleComplex*)UUhat[0].begin(), (cuDoubleComplex*)UUhat[1].begin(), (cuDoubleComplex*)UUhat[2].begin());
      #endif
    }

    for(Index kk = 0; kk < gi.r; kk ++){
      fft->backward(UUhat[0].begin()+kk*gi.dxxh_mult,UU[0].begin()+kk*gi.dxx_mult,UU[0].sl);
      fft->backward(UUhat[1].begin()+kk*gi.dxxh_mult,UU[1].begin()+kk*gi.dxx_mult,UU[1].sl);
      fft->backward(UUhat[2].begin()+kk*gi.dxxh_mult,UU[2].begin()+kk*gi.dxx_mult,UU[2].sl);
    }

    blas->matmul_transb(UU[0],C1[0],tmpX2);
    blas->matmul_transb(UU[1],C1[1],UU[0]);
    tmpX2 += UU[0];
    blas->matmul_transb(UU[2],C1[2],UU[0]);
    tmpX2 += UU[0];

    ptw_mult_row(U,ef[0],UU[0]);
    blas->matmul_transb(UU[0],C2[0],UU[1]);
    ptw_mult_row(U,ef[1],UU[0]);
    blas->matmul_transb(UU[0],C2[1],UU[2]);
    UU[1] += UU[2];
    ptw_mult_row(U,ef[2],UU[0]);
    blas->matmul_transb(UU[0],C2[2],out);
    addsub_rhs(out,UU[1],tmpX2); //out += UU[1]; //out -= tmpX2;
  }


  void rk4_K(mat& U, Index n, double tau, array<vec,3>& ef, const array<mat,3>& C1, const array<mat,3>& C2){
    // Input overwritten
    for(Index i = 0; i < n; i++){
      rk4_K_rhs(U, ef, C1, C2, KK[0]);
      setmultadd_rk4(tmpX,U,tau/2.0,KK[0]); //tmpX = KK[0]; //tmpX *= (tau/2.0); //tmpX += U;
      rk4_K_rhs(tmpX, ef, C1, C2, KK[1]);
      setmultadd_rk4(tmpX,U,tau/2.0,KK[1]); //tmpX = KK[1]; //tmpX *= (tau/2.0); //tmpX += U;
      rk4_K_rhs(tmpX, ef, C1, C2, KK[2]);
      setmultadd_rk4(tmpX,U,tau,KK[2]); //tmpX = KK[2]; //tmpX *= tau; //tmpX += U;
      rk4_K_rhs(tmpX, ef, C1, C2, KK[3]);
      finstage_rk4(U,KK[0],KK[1],KK[2],KK[3],tau);
    }
  }

  void operator()(double tau, mat& K, array<vec,3>& ef, const array<mat,3>& C1, const array<mat,3>& C2, Index nsteps_int=1) {

    if(fft == nullptr)
      fft = make_unique_ptr<fft3d<2>>(gi.N_xx, K, Uhat, true);
    rk4_K(K, nsteps_int, tau/nsteps_int, ef, C1, C2);
  }
      
  void update_info(Index nr){
    gi.update_rank(nr);
    Uhat.update_shape({gi.dxxh_mult,gi.r});
    for(int i=0; i<3; i++){
      UUhat[i].update_shape({gi.dxxh_mult,gi.r});
      UU[i].update_shape({gi.dxx_mult,gi.r});
      KK[i].update_shape({gi.dxx_mult,gi.r});
    }
    KK[3].update_shape({gi.dxx_mult,gi.r});
    
    tmpX.update_shape({gi.dxx_mult,gi.r});
    tmpX2.update_shape({gi.dxx_mult,gi.r});

  }


private:
  stloc sl;
  grid_info_reserve<3> gi;
  const blas_ops* blas;

  std::unique_ptr<fft3d<2>> fft;
  cmat Uhat;
  array<cmat,3> UUhat;
  array<mat,3> UU;
  array<mat,4> KK;
  mat tmpX, tmpX2;
  std::unique_ptr<vec> d_lim_xx;
};

struct PS_S_step_adapt {

  PS_S_step_adapt(stloc _sl, grid_info_reserve<3> _gi, const blas_ops* _blas)
    : sl(_sl), gi(_gi), blas(_blas), tmpSS(_sl), tmpS(_sl), tmpS2(_sl) {

      tmpSS.resize({gi.r,gi.r});
      tmpS.resize({gi.r,gi.r});
      tmpS2.resize({gi.r,gi.r});
      SS = create_rk4_array({gi.r,gi.r}, sl);
  }

  void rk4_S_rhs(mat& U, const array<mat,3>& C1, const array<mat,3>& C2, const array<mat,3>& D1, const array<mat,3>& D2, mat& out){

    blas->matmul(D2[0],U,tmpS);
    blas->matmul_transb(tmpS,C1[0],out);
    blas->matmul(D2[1],U,tmpS);
    blas->matmul_transb(tmpS,C1[1],tmpS2);
    out += tmpS2;
    blas->matmul(D2[2],U,tmpS);
    blas->matmul_transb(tmpS,C1[2],tmpS2);
    out += tmpS2;
    blas->matmul(D1[0],U,tmpS);
    blas->matmul_transb(tmpS,C2[0],tmpS2);
    out -= tmpS2;
    blas->matmul(D1[1],U,tmpS);
    blas->matmul_transb(tmpS,C2[1],tmpS2);
    out -= tmpS2;
    blas->matmul(D1[2],U,tmpS);
    blas->matmul_transb(tmpS,C2[2],tmpS2);
    out -= tmpS2;
  }

  void rk4_S(mat& U, Index n, double tau, const array<mat,3>& C1, const array<mat,3>& C2, const array<mat,3>& D1,const array<mat,3>& D2){
    //Input overwritten
    for(Index i = 0; i < n; i++){
      rk4_S_rhs(U, C1, C2, D1, D2, SS[0]);
      setmultadd_rk4(tmpSS,U,tau/2.0, SS[0]);
      rk4_S_rhs(tmpSS, C1, C2, D1, D2, SS[1]);
      setmultadd_rk4(tmpSS,U,tau/2.0,SS[1]);
      rk4_S_rhs(tmpSS, C1, C2, D1, D2, SS[2]);
      setmultadd_rk4(tmpSS,U,tau,SS[2]);
      rk4_S_rhs(tmpSS, C1, C2, D1, D2, SS[3]);
      finstage_rk4(U,SS[0],SS[1],SS[2],SS[3],tau);
    }
  }

  void operator()(double tau, mat& S, const array<mat,3>& C1, const array<mat,3>& C2, const array<mat,3>& D1, const array<mat,3>& D2, Index nsteps_int=1) {

    rk4_S(S, nsteps_int, tau/nsteps_int, C1, C2, D1, D2);
  }

  void update_info(Index nr){
    gi.update_rank(nr);
    tmpSS.resize({gi.r,gi.r});
    tmpS.resize({gi.r,gi.r});
    tmpS2.resize({gi.r,gi.r});
    for(int i=0; i<4; i++){
      SS[i].resize({gi.r,gi.r});
    }

  }

private:
  stloc sl;
  grid_info_reserve<3> gi;
  const blas_ops* blas;

  mat tmpSS, tmpS, tmpS2;
  array<mat,4> SS;

};

struct PS_L_step_adapt {

  PS_L_step_adapt(stloc _sl, grid_info_reserve<3> _gi, const blas_ops* _blas)
    : sl(_sl), gi(_gi), blas(_blas), fft(nullptr), Vhat(_sl), tmpV(_sl), tmpV2(_sl) {

      Vhat.reserve({gi.dvvh_mult,gi.rmax},{gi.dvvh_mult,gi.r});
      VVhat = initialize_cmat_array(sl);
      VV = initialize_mat_array(sl);
      LL = initialize_rk4_array(sl);

      for(int i = 0; i < 3; i++){
        VVhat[i].reserve({gi.dvvh_mult,gi.rmax},{gi.dvvh_mult,gi.r});
        VV[i].reserve({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r});
        LL[i].reserve({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r});
      }
      LL[3].reserve({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r});

      tmpV.reserve({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r});
      tmpV2.reserve({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r});

      v = create_vec_array(gi.dvv_mult,sl);
      array<vec,3> h_v = create_vec_array(gi.dvv_mult, stloc::host);
      componentwise_vec_omp(gi.N_vv, [this, &h_v](Index idx, mind<3> i) {
        h_v[0](idx) = gi.v(0, i[0]);
        h_v[1](idx) = gi.v(1, i[1]);
        h_v[2](idx) = gi.v(2, i[2]);
      });
      v = h_v;

      #ifdef __CUDA__
      if(sl == stloc::device) {
        d_lim_vv = make_unique_ptr<vec>(array<Index,1>({6}), stloc::device);
        cudaMemcpy(d_lim_vv->data(), gi.lim_vv.data(), 6*sizeof(double), cudaMemcpyHostToDevice);
      }
      #endif
    }


  void rk4_L_rhs(mat& V, array<vec,3>& v, const array<mat,3>& D1, const array<mat,3>& D2, mat& out){

    // Perform needed derivatives in Fourier space
    // TODO: if needed we could work on a single column
    for(Index kk = 0; kk < gi.r; kk ++){
      fft->forward(V.begin()+kk*gi.dvv_mult,Vhat.begin()+kk*gi.dvvh_mult,V.sl);
    }
    double ncvv = 1.0 / double(gi.dvv_mult);
    if(sl == stloc::host){
      componentwise_mat_fourier_omp(gi.r, gi.N_vv, [this, ncvv](Index idx, mind<3> i, Index rr) {
      Index mult_j = freq(i[1], gi.N_vv[1]);
      Index mult_k = freq(i[2], gi.N_vv[2]);
      complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[1]-gi.lim_vv[0])*i[0]);
      complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[3]-gi.lim_vv[2])*mult_j);
      complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[5]-gi.lim_vv[4])*mult_k);

      VVhat[0](idx,rr) = Vhat(idx,rr) * lambdav * ncvv;
      VVhat[1](idx,rr) = Vhat(idx,rr) * lambdaw * ncvv ;
      VVhat[2](idx,rr) = Vhat(idx,rr) * lambdau * ncvv ;
      });
    } else {
      #ifdef __CUDA__
        ptw_mult_row_cplx_fourier_3d<<<(gi.dvvh_mult*gi.r+n_threads-1)/n_threads,n_threads>>>(gi.dvvh_mult*gi.r, gi.N_vv[0]/2+1, gi.N_vv[1], gi.N_vv[2], (cuDoubleComplex*)Vhat.begin(), d_lim_vv->data(), ncvv, (cuDoubleComplex*)VVhat[0].begin(), (cuDoubleComplex*)VVhat[1].begin(), (cuDoubleComplex*)VVhat[2].begin());
      #endif
    }

    for(Index kk = 0; kk < gi.r; kk ++){
      fft->backward(VVhat[0].begin()+kk*gi.dvvh_mult,VV[0].begin()+kk*gi.dvv_mult,VV[0].sl);
      fft->backward(VVhat[1].begin()+kk*gi.dvvh_mult,VV[1].begin()+kk*gi.dvv_mult,VV[1].sl);
      fft->backward(VVhat[2].begin()+kk*gi.dvvh_mult,VV[2].begin()+kk*gi.dvv_mult,VV[2].sl);
    }

    blas->matmul_transb(VV[0],D1[0],out);
    blas->matmul_transb(VV[1],D1[1],VV[0]);
    out += VV[0];
    blas->matmul_transb(VV[2],D1[2],VV[0]);
    out += VV[0];

    ptw_mult_row(V,v[0],VV[0]);
    blas->matmul_transb(VV[0],D2[0],VV[1]);
    out -= VV[1];
    ptw_mult_row(V,v[1],VV[0]);
    blas->matmul_transb(VV[0],D2[1],VV[2]);
    out -= VV[2];
    ptw_mult_row(V,v[2],VV[0]);
    blas->matmul_transb(VV[0],D2[2],tmpV2);
    out -= tmpV2;
  }


  void rk4_L(mat& V, Index n, double tau, array<vec,3>& v, const array<mat,3>& D1, const array<mat,3>& D2){
    // Input overwritten
    for(Index i = 0; i < n; i++){
      rk4_L_rhs(V, v, D1, D2, LL[0]);
      setmultadd_rk4(tmpV,V,tau/2.0,LL[0]);
      rk4_L_rhs(tmpV, v, D1, D2, LL[1]);
      setmultadd_rk4(tmpV,V,tau/2.0,LL[1]);
      rk4_L_rhs(tmpV, v, D1, D2, LL[2]);
      setmultadd_rk4(tmpV,V,tau,LL[2]);
      rk4_L_rhs(tmpV, v, D1, D2, LL[3]);
      finstage_rk4(V,LL[0],LL[1],LL[2],LL[3],tau);
    }
  }

  void operator()(double tau, mat& L, const array<mat,3>& D1, const array<mat,3>& D2, Index nsteps_int=1) {

    if(fft == nullptr)
      fft = make_unique_ptr<fft3d<2>>(gi.N_vv, L, Vhat, true);
    rk4_L(L, nsteps_int, tau/nsteps_int, v, D1, D2);
  }

  array<vec,3> get_v(){
    return v;
  }
  
  void update_info(Index nr){
    gi.update_rank(nr);
    Vhat.update_shape({gi.dvvh_mult,gi.r});
    for(int i=0; i<3; i++){
      VVhat[i].update_shape({gi.dvvh_mult,gi.r});
      VV[i].update_shape({gi.dvv_mult,gi.r});
      LL[i].update_shape({gi.dvv_mult,gi.r});
    }
    LL[3].update_shape({gi.dvv_mult,gi.r});
    
    tmpV.update_shape({gi.dvv_mult,gi.r});
    tmpV2.update_shape({gi.dvv_mult,gi.r});

  }


private:
  stloc sl;
  grid_info_reserve<3> gi;
  const blas_ops* blas;

  std::unique_ptr<fft3d<2>> fft;
  cmat Vhat;
  array<cmat,3> VVhat;
  array<mat,3> VV;
  array<mat,4> LL;
  mat tmpV, tmpV2;
  array<vec,3> v;
  std::unique_ptr<vec> d_lim_vv;
};

struct BUG_S_step_adapt {

  BUG_S_step_adapt(stloc _sl, grid_info_reserve<3> _gi, const blas_ops* _blas, int _fact )
    : sl(_sl), gi(_gi), blas(_blas), tmpSS(_sl), tmpS(_sl), tmpS2(_sl), fact(_fact) {
      
      tmpSS.resize({fact*gi.r,fact*gi.r});
      tmpS.resize({fact*gi.r,fact*gi.r});
      tmpS2.resize({fact*gi.r,fact*gi.r});
      SS = create_rk4_array({fact*gi.r,fact*gi.r}, sl);
  }

  void rk4_S_rhs(mat& U, const array<mat,3>& C1, const array<mat,3>& C2, const array<mat,3>& D1, const array<mat,3>& D2, mat& out){

    blas->matmul(D2[0],U,tmpS);
    blas->matmul_transb(tmpS,C1[0],out);
    blas->matmul(D2[1],U,tmpS);
    blas->matmul_transb(tmpS,C1[1],tmpS2);
    out += tmpS2;
    blas->matmul(D2[2],U,tmpS);
    blas->matmul_transb(tmpS,C1[2],tmpS2);
    out += tmpS2;
    blas->matmul(D1[0],U,tmpS);
    blas->matmul_transb(tmpS,C2[0],tmpS2);
    out -= tmpS2;
    blas->matmul(D1[1],U,tmpS);
    blas->matmul_transb(tmpS,C2[1],tmpS2);
    out -= tmpS2;
    blas->matmul(D1[2],U,tmpS);
    blas->matmul_transb(tmpS,C2[2],tmpS2);
    out -= tmpS2;
    out *= -1;
  }

  void rk4_S(mat& U, Index n, double tau, const array<mat,3>& C1, const array<mat,3>& C2, const array<mat,3>& D1,const array<mat,3>& D2){
    //Input overwritten
    for(Index i = 0; i < n; i++){
      rk4_S_rhs(U, C1, C2, D1, D2, SS[0]);
      setmultadd_rk4(tmpSS,U,tau/2.0, SS[0]);
      rk4_S_rhs(tmpSS, C1, C2, D1, D2, SS[1]);
      setmultadd_rk4(tmpSS,U,tau/2.0,SS[1]);
      rk4_S_rhs(tmpSS, C1, C2, D1, D2, SS[2]);
      setmultadd_rk4(tmpSS,U,tau,SS[2]);
      rk4_S_rhs(tmpSS, C1, C2, D1, D2, SS[3]);
      finstage_rk4(U,SS[0],SS[1],SS[2],SS[3],tau);
    }
  }

  void operator()(double tau, mat& S, const array<mat,3>& C1, const array<mat,3>& C2, const array<mat,3>& D1, const array<mat,3>& D2, const blas_ops& blas, Index nsteps_int=1) {

    rk4_S(S, nsteps_int, tau/nsteps_int, C1, C2, D1, D2);
  }

  void update_info(Index nr){
    gi.update_rank(nr);
    tmpSS.resize({fact*gi.r,fact*gi.r});
    tmpS.resize({fact*gi.r,fact*gi.r});
    tmpS2.resize({fact*gi.r,fact*gi.r});
    for(int i=0; i<4; i++){
      SS[i].resize({fact*gi.r,fact*gi.r});
    }

  }

private:
  stloc sl;
  grid_info_reserve<3> gi;
  const blas_ops* blas;

  mat tmpSS, tmpS, tmpS2;
  int fact;
  array<mat,4> SS;

};

void mgs_orthcol_cpu(multi_array<double,2>& X, double w) {

  array<Index,2> dims = X.shape();
  Index rk = dims[1];

  std::default_random_engine generator(1234);
  std::normal_distribution<double> distribution(0.0,1.0);
  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index i = 0; i < dims[0]; i++){
    X(i,rk-1) = distribution(generator);
  }
  for(Index k=0;k<(rk-1);k++) {
    cblas_daxpy(dims[0], -w*cblas_ddot(dims[0],&X(0,rk-1),1,&X(0,k),1), X.extract({k}), 1, X.extract({rk-1}),1);
  }

      cblas_dscal(dims[0],1.0/sqrt(w*cblas_ddot(dims[0],&X(0,rk-1),1,&X(0,rk-1),1)),X.extract({rk-1}),1);
}


#ifdef __CUDA__
  void mgs_orthcol_gpu(multi_array<double,2>& X, double w, const blas_ops& blas) {
    array<Index,2> dims = X.shape();
    Index n = dims[0];
    Index rk = dims[1];
    double* r;
    cudaMalloc((void**)&r,sizeof(double));

    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,1234);

    curandGenerateNormalDouble(gen,&X(0,rk-1),n, 0.0, 1.0);
  
    for(Index k=0;k<(rk-1);k++) {
      cublasDdot(blas.handle_devres, n, &X(0,rk-1), 1, &X(0,k), 1, r);
      scale_unique<<<1,1>>>(r,w);
      cudaDeviceSynchronize();
      dmaxpy<<<(n+n_threads-1)/n_threads,n_threads>>>(n, r, &X(0,k), &X(0,rk-1));
      cudaDeviceSynchronize();
    }
    cublasDdot(blas.handle_devres, n, &X(0,rk-1), 1, &X(0,rk-1), 1, r);
    scale_sqrt_unique<<<1,1>>>(r,w);
    cudaDeviceSynchronize();
    ptw_div_gs<<<(n+n_threads-1)/n_threads,n_threads>>>(n, &X(0,rk-1), r);
    cudaDeviceSynchronize();
    curandDestroyGenerator(gen);
  }

#endif

double electric_energy(array<vec,3>& E, grid_info_reserve<3>& gi, const blas_ops& blas){
  double ee = 0.0;
  if(E[0].sl == stloc::host){
    #ifdef __OPENMP__
    #pragma omp parallel for reduction(+:ee)
    #endif
    for(Index ii = 0; ii < gi.dxx_mult; ii++){
      ee += 0.5*(pow(E[0](ii),2)+pow(E[1](ii),2)+pow(E[2](ii),2))*gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2];
    }
  } else {
    #ifdef __CUDA__
      double* d_el_energy;
      cudaMalloc(&d_el_energy, sizeof(double)*3);
      cublasDdot (blas.handle_devres, E[0].num_elements(), E[0].begin(), 1, E[0].begin(), 1, d_el_energy);
      cublasDdot (blas.handle_devres, E[1].num_elements(), E[1].begin(), 1, E[1].begin(), 1, d_el_energy+1);
      cublasDdot (blas.handle_devres, E[2].num_elements(), E[2].begin(), 1, E[2].begin(), 1, d_el_energy+2);
      cudaDeviceSynchronize();
      ptw_sum<<<1,1>>>(1,d_el_energy,d_el_energy+1);
      cudaDeviceSynchronize();
      ptw_sum<<<1,1>>>(1,d_el_energy,d_el_energy+2);
      cudaDeviceSynchronize();
      scale_unique<<<1,1>>>(d_el_energy,0.5*gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);

      cudaMemcpy(&ee,d_el_energy,sizeof(double),cudaMemcpyDeviceToHost);
      cudaFree(d_el_energy);
    #endif
  }

  return ee;
}

double norm_EF(array<vec,3>& E, grid_info_reserve<3>& gi, const blas_ops& blas){
  double L2norms2[3];

  if(E[0].sl == stloc::host){
    for(int i = 0; i < 3; i++){
      double norm_tmp = 0.0;
      #ifdef __OPENMP__
      #pragma omp parallel for reduction(+:norm_tmp)
      #endif
      for(Index ii = 0; ii < gi.dxx_mult; ii++){
        norm_tmp += E[i](ii)*E[i](ii);
      }
      L2norms2[i] = norm_tmp;
    }
  } else {
    #ifdef __CUDA__
      double* d_L2norms2;
      cudaMalloc(&d_L2norms2, sizeof(double)*3);
      cublasDdot (blas.handle_devres, E[0].num_elements(), E[0].begin(), 1, E[0].begin(), 1, d_L2norms2);
      cublasDdot (blas.handle_devres, E[1].num_elements(), E[1].begin(), 1, E[1].begin(), 1, d_L2norms2+1);
      cublasDdot (blas.handle_devres, E[2].num_elements(), E[2].begin(), 1, E[2].begin(), 1, d_L2norms2+2);
      cudaDeviceSynchronize();
      cudaMemcpy(L2norms2,d_L2norms2,sizeof(double)*3,cudaMemcpyDeviceToHost);
      cudaFree(d_L2norms2);
    #endif
  }

  return sqrt((L2norms2[0]+L2norms2[1]+L2norms2[2])*gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);
}

double norm_err_EF(array<vec,3>& E1, array<vec,3>& E2, grid_info_reserve<3>& gi, const blas_ops& blas){
  double L2norms2[3];
  if(E1[0].sl == stloc::host){
    for(int i = 0; i < 3; i++){
      double norm_tmp = 0.0;
      #ifdef __OPENMP__
      #pragma omp parallel for reduction(+:norm_tmp)
      #endif
      for(Index ii = 0; ii < gi.dxx_mult; ii++){
        norm_tmp += (E1[i](ii)-E2[i](ii))*(E1[i](ii)-E2[i](ii));
      }
      L2norms2[i] = norm_tmp;
    }
  } else {
    #ifdef __CUDA__
      double* d_L2norms2;
      cudaMalloc(&d_L2norms2, sizeof(double)*3);
      for(int i = 0; i < 3; i++){
        E1[i] -= E2[i];
        cublasDdot (blas.handle_devres, E1[i].num_elements(), E1[i].begin(), 1, E1[i].begin(), 1, d_L2norms2+i);
      }
      cudaMemcpy(L2norms2,d_L2norms2,sizeof(double)*3,cudaMemcpyDeviceToHost);
      cudaFree(d_L2norms2);
    #endif
  }

  return sqrt((L2norms2[0]+L2norms2[1]+L2norms2[2])*gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);
}

void extract_rows(mat& A, Index init, Index fin, mat& B){
  Index ncols = A.shape()[1];
  for(Index i=0; i<ncols; i++){
    if (A.sl == stloc::host){
      std::copy(&A({init,i}),&A({fin,i})+1,&B({0,i}));
    } else{
      #ifdef __CUDA__
      cudaMemcpy(&B({0,i}), &A({init,i}), sizeof(double)*(fin-init+1),cudaMemcpyDeviceToDevice);
      #endif
    }
  }

  return;

}

void PS_adapt(double final_time, double tau, int nsteps_int, grid_info_reserve<3>& gi, vector<const double*> X0, vector<const double*> V0, double tol_inc, double tol_inc_rel, double tol_dec, double tol_dec_rel, Index min_r, Index max_r, string ec, Index snapshots, const blas_ops& blas){

  gt::start("Initialization");
  stloc sl = (CPU) ? stloc::host : stloc::device;
  string gstr;

  orthogonalize gs(&blas);

  // Initialization
  lr2<double> lr_sol(gi.r,max_r,{gi.dxx_mult,gi.dvv_mult}, sl);

  if(sl == stloc::host) {
    initialize(lr_sol, X0, V0, gi.h_xx, gi.h_vv, blas);
    gstr = "cpu";
  } else {
    lr2<double> h_lr_sol(gi.r,max_r,{gi.dxx_mult,gi.dvv_mult}, stloc::host);
    initialize(h_lr_sol, X0, V0, gi.h_xx, gi.h_vv, blas);
    lr_sol = h_lr_sol;
    gstr = "gpu";
  }
  ofstream el_energyf("evolution_ps_"+ec+"_"+gstr+".data");
  ofstream contf("contf_ps_"+ec+"_"+gstr+".data");
  el_energyf.precision(16);
  double t = 0.0;
  Index n_steps = ceil(final_time/tau);


  vector<int> h_rank(n_steps);

  array<vec,3> E = create_vec_array(gi.dxx_mult, sl);
  array<vec,3> Etmp = create_vec_array(gi.dxx_mult, sl);
  array<vec,3> Etmp2 = create_vec_array(gi.dxx_mult, sl);

  PS_K_step_adapt K_step_rk4(sl, gi, &blas);
  PS_S_step_adapt S_step_rk4(sl, gi, &blas);
  PS_L_step_adapt L_step_rk4(sl, gi, &blas);

  mat Xn({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r}, sl);
  mat Sn({gi.r,gi.r}, sl);
  mat Vn({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r}, sl);

  electric_field efield(sl, gi);

  coeff_C compute_C(sl, gi);
  coeff_D compute_D(sl, gi);

  array<mat, 3> C1 = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> C2 = create_mat_array({gi.r,gi.r}, sl);

  array<mat, 3> D1 = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> D2 = create_mat_array({gi.r,gi.r}, sl);

  // needed for error control
  mat Kad({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r}, sl);
  mat UUs({gi.r,gi.r}, sl);
  mat VVs({gi.r,gi.r}, sl);
  vec sigma({gi.r}, sl);
  mat tmps({gi.r,gi.r}, sl);

  gt::stop("Initialization");

  Index kk = 1;
  cout.precision(8);
  gt::start("Main loop");
  while(kk<=n_steps){

    cout << "Step " << kk << " of " << n_steps << endl;
  
    gt::start("Time integration");

    h_rank[kk-1] = (int)gi.r;

    gt::start("K step");
    // Compute K
    blas.matmul(lr_sol.X,lr_sol.S,Xn); // Xn is K
    gt::stop("K step");

    gt::start("Electric field");
    // Electric field
    efield(Xn, lr_sol.V, E, blas);
    gt::stop("Electric field");
    gt::start("Electric energy");
    double el_energy = electric_energy(E, gi, blas);
    gt::stop("Electric energy");

    cout << el_energy << endl;

    // ---- K step ----
    gt::start("C coeffs");
    compute_C(lr_sol.V, C1, C2, blas);
    gt::stop("C coeffs");
    gt::start("K step");
    K_step_rk4(tau, Xn, E, C1, C2, nsteps_int);
    gt::start("gs K step");
    gs(Xn, Sn, gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]); // Xn the new X
    gt::stop("gs K step");
    gt::stop("K step");
 
    // ---- S step ----
    gt::start("D coeffs");
    compute_D(Xn, E, D1, D2, blas);
    gt::stop("D coeffs");
    gt::start("S step");
    S_step_rk4(tau, Sn, C1, C2, D1, D2, nsteps_int);
    gt::stop("S step");
    
    // ---- L step ----
    gt::start("L step");
    blas.matmul_transb(lr_sol.V,Sn,Vn); // Vn is L
    L_step_rk4(tau, Vn, D1, D2, nsteps_int);
    gs(Vn, Sn, gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
    transpose_inplace(Sn);
    gt::stop("L step");
    
    gt::stop("Time integration");
    
    gt::start("Rank choice");

    if (ec == "f"){
      gt::start("SVD decomposition");
      svd(Sn, UUs, VVs, sigma, blas);
      gt::stop("SVD decomposition");
      double svr=0.0;
      if(Sn.sl == stloc::host){
        svr = sigma(gi.r-1);
      } else {
        #ifdef __CUDA__
          cudaMemcpy(&svr, &sigma(gi.r-1), sizeof(double),cudaMemcpyDeviceToHost);
        #endif
      }

      if (svr >= tol_inc){
        if (gi.r == max_r){
          contf << "j" << endl;
          cout << "Should reject and increase rank but max rank reached. Proceeding keeping max rank." << endl;

          gt::start("Reject but max rank: swap");
          lr_sol.X.swap(Xn);
          lr_sol.V.swap(Vn);
          lr_sol.S.swap(Sn);
          gt::stop("Reject but max rank: swap");
 
          el_energyf << t << " " << el_energy << endl;
          t += tau;

          kk = kk + 1;
        } else {
          contf << "r" << endl;
          cout << "Rejected step, increasing rank by one." << endl;

          gt::start("Reject: increase rank");
          // Do all the updates
          gt::start("Reject: increase rank (updates)");

          lr_sol.S.swap(Sn);

          gi.update_rank(gi.r+1);
          lr_sol.update_info(gi.r);
          K_step_rk4.update_info(gi.r);
          S_step_rk4.update_info(gi.r);
          L_step_rk4.update_info(gi.r);

          Xn.update_shape({gi.dxx_mult,gi.r});
          Vn.update_shape({gi.dvv_mult,gi.r});
          gt::stop("Reject: increase rank (updates)");

          gt::start("Reject: increase rank (gram schmidt)");
          if(lr_sol.X.sl == stloc::host){
            mgs_orthcol_cpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);
            mgs_orthcol_cpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
          } else {
            #ifdef __CUDA__
              mgs_orthcol_gpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2],blas);
              mgs_orthcol_gpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2],blas);
            #endif
          }
          gt::stop("Reject: increase rank (gram schmidt)");

          gt::start("Reject: increase rank (some resizes rxr)");

          // This could directly become a kernel, but s times s so we'll see
          #ifdef __OPENMP__
          #pragma omp parallel for
          #endif
          for(Index i=0; i < lr_sol.S.num_elements(); i++){
              Index idx_r = i%gi.r;
              Index idx_c = i/gi.r;
              if((idx_r == (gi.r-1)) || (idx_c == (gi.r-1))){
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = 0.0;
                } else {
                  #ifdef __CUDA__
                    double ZERO = 0.0;
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
                  #endif
                }
              }
              else {
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                } else {
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
              }
          }
          Sn.resize({gi.r,gi.r});
          
          for(int ii = 0; ii < 3; ii++){
            C1[ii].resize({gi.r,gi.r});
            C2[ii].resize({gi.r,gi.r});
            D1[ii].resize({gi.r,gi.r});
            D2[ii].resize({gi.r,gi.r});
          }
          UUs.resize({gi.r,gi.r});
          VVs.resize({gi.r,gi.r});
          tmps.resize({gi.r,gi.r});
          sigma.resize({gi.r});

          Kad.update_shape({gi.dxx_mult,gi.r});
          efield.update_info(gi.r);
          compute_C.update_info(gi.r);
          compute_D.update_info(gi.r);
          gt::stop("Reject: increase rank (some resizes rxr)");
          gt::stop("Reject: increase rank");
        }
      } else if (svr <= tol_dec){
          if (gi.r == min_r){
            contf << "m" << endl;
            cout << "Accepted step, should decrease rank but min rank reached. Proceeding keeping min rank." << endl;

            gt::start("Accept but min rank: swap");
            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);
            lr_sol.S.swap(Sn);

            gt::stop("Accept but min rank: swap");

          } else {
            contf << "a" << endl;
            cout << "Accepted step, decreasing rank by one." << endl;

            gt::start("Accept: decrease rank");

            // Do all the updates
            gi.update_rank(gi.r-1);
            lr_sol.update_info(gi.r);
            K_step_rk4.update_info(gi.r);
            S_step_rk4.update_info(gi.r);
            L_step_rk4.update_info(gi.r);

            Xn.update_shape({gi.dxx_mult,gi.r});
            Vn.update_shape({gi.dvv_mult,gi.r});
            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);

            #ifdef __OPENMP__
            #pragma omp parallel for
            #endif
            for(Index i=0; i < lr_sol.S.num_elements(); i++){
                Index idx_r = i%gi.r;
                Index idx_c = i/gi.r;
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                }else{
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
            }
            Sn.resize({gi.r,gi.r});

            for(int ii = 0; ii < 3; ii++){
              C1[ii].resize({gi.r,gi.r});
              C2[ii].resize({gi.r,gi.r});
              D1[ii].resize({gi.r,gi.r});
              D2[ii].resize({gi.r,gi.r});
            }
            UUs.resize({gi.r,gi.r});
            VVs.resize({gi.r,gi.r});
            tmps.resize({gi.r,gi.r});
            sigma.resize({gi.r});

            Kad.update_shape({gi.dxx_mult,gi.r});
            efield.update_info(gi.r);
            compute_C.update_info(gi.r);
            compute_D.update_info(gi.r);
            gt::stop("Accept: decrease rank");

          }
          el_energyf << t << " " << el_energy << endl;
          t += tau;
          kk = kk + 1;
      } else{
        contf << "s" << endl;
        cout << "Accepted step, keeping same rank." << endl;
        gt::start("Accept keep rank: swap");
        lr_sol.X.swap(Xn);
        lr_sol.V.swap(Vn);
        lr_sol.S.swap(Sn);
        gt::stop("Accept keep rank: swap");

        el_energyf << t << " " << el_energy << endl;
        t += tau;
        kk = kk + 1;
      }
    }else if (ec == "ef"){
      gt::start("NEW el field (matmul, efield, ee)");
      blas.matmul(Xn,Sn,Kad);
      efield(Kad, Vn, Etmp, blas);
      gt::stop("NEW el field (matmul, efield, ee)");
      gt::start("SVD decomposition");
      svd(Sn, UUs, VVs, sigma, blas);
      gt::stop("SVD decomposition");
      if(Sn.sl == stloc::host){
        sigma(gi.r-1) = 0.0;
      } else {
        #ifdef __CUDA__
          double ZERO = 0.0;
          cudaMemcpy(&sigma(gi.r-1),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
        #endif
      }

      gt::start("CUT el field (pmr, matmul, efield, ee)");
      //TODO: can be optimized, but it's r times r
      transpose_inplace(VVs);
      ptw_mult_row(VVs,sigma,tmps);
      blas.matmul(UUs,tmps,VVs);

      blas.matmul(Xn,VVs,Kad);
      efield(Kad, Vn, Etmp2, blas);

      gt::stop("CUT el field (pmr, matmul, efield, ee)");

      double norm_ef = norm_EF(Etmp, gi, blas);
      double err_ef = norm_err_EF(Etmp, Etmp2, gi, blas);

      if (err_ef >= (tol_inc+norm_ef*tol_inc_rel)){
        if (gi.r == max_r){
          contf << "j" << endl;
          cout << "Should reject and increase rank but max rank reached. Proceeding keeping max rank." << endl;

          gt::start("Reject but max rank: swap");
          lr_sol.X.swap(Xn);
          lr_sol.V.swap(Vn);
          lr_sol.S.swap(Sn);
          gt::stop("Reject but max rank: swap");
 
          el_energyf << t << " " << el_energy << endl;
          t += tau;

          kk = kk + 1;
        } else {
          contf << "r" << endl;
          cout << "Rejected step, increasing rank by one." << endl;

          gt::start("Reject: increase rank");
          // Do all the updates
          gt::start("Reject: increase rank (updates)");
          
          lr_sol.S.swap(Sn);

          gi.update_rank(gi.r+1);
          lr_sol.update_info(gi.r);
          K_step_rk4.update_info(gi.r);
          S_step_rk4.update_info(gi.r);
          L_step_rk4.update_info(gi.r);

          Xn.update_shape({gi.dxx_mult,gi.r});
          Vn.update_shape({gi.dvv_mult,gi.r});
          gt::stop("Reject: increase rank (updates)");

          gt::start("Reject: increase rank (gram schmidt)");
          if(lr_sol.X.sl == stloc::host){
            mgs_orthcol_cpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);
            mgs_orthcol_cpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
          } else {
            #ifdef __CUDA__
              mgs_orthcol_gpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2],blas);
              mgs_orthcol_gpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2],blas);
            #endif
          }
          gt::stop("Reject: increase rank (gram schmidt)");

          gt::start("Reject: increase rank (some resizes rxr)");

          // This could directly become a kernel, but r times r so we'll see
          #ifdef __OPENMP__
          #pragma omp parallel for
          #endif
          for(Index i=0; i < lr_sol.S.num_elements(); i++){
              Index idx_r = i%gi.r;
              Index idx_c = i/gi.r;
              if((idx_r == (gi.r-1)) || (idx_c == (gi.r-1))){
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = 0.0;
                } else {
                  #ifdef __CUDA__
                    double ZERO = 0.0;
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
                  #endif
                }
              } else {
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                } else {
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
              }
          }
          Sn.resize({gi.r,gi.r});

          for(int ii = 0; ii < 3; ii++){
            C1[ii].resize({gi.r,gi.r});
            C2[ii].resize({gi.r,gi.r});
            D1[ii].resize({gi.r,gi.r});
            D2[ii].resize({gi.r,gi.r});
          }
          UUs.resize({gi.r,gi.r});
          VVs.resize({gi.r,gi.r});
          tmps.resize({gi.r,gi.r});
          sigma.resize({gi.r});

          Kad.update_shape({gi.dxx_mult,gi.r});
          efield.update_info(gi.r);
          compute_C.update_info(gi.r);
          compute_D.update_info(gi.r);
          gt::stop("Reject: increase rank (some resizes rxr)");
          gt::stop("Reject: increase rank");
        }
      } else if (err_ef <= (tol_dec+norm_ef*tol_dec_rel)){
          if (gi.r == min_r){
            contf << "m" << endl;
            cout << "Accepted step, should decrease rank but min rank reached. Proceeding keeping min rank." << endl;

            gt::start("Accept but min rank: swap");
            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);
            lr_sol.S.swap(Sn);
            gt::stop("Accept but min rank: swap");

          } else {
            contf << "a" << endl;
            cout << "Accepted step, decreasing rank by one." << endl;

            gt::start("Accept: decrease rank");

            // Do all the updates
            gi.update_rank(gi.r-1);
            lr_sol.update_info(gi.r);
            K_step_rk4.update_info(gi.r);
            S_step_rk4.update_info(gi.r);
            L_step_rk4.update_info(gi.r);

            Xn.update_shape({gi.dxx_mult,gi.r});
            Vn.update_shape({gi.dvv_mult,gi.r});
            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);

            #ifdef __OPENMP__
            #pragma omp parallel for
            #endif
            for(Index i=0; i < lr_sol.S.num_elements(); i++){
                Index idx_r = i%gi.r;
                Index idx_c = i/gi.r;
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                }else{
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
            }
            Sn.resize({gi.r,gi.r});

            for(int ii = 0; ii < 3; ii++){
              C1[ii].resize({gi.r,gi.r});
              C2[ii].resize({gi.r,gi.r});
              D1[ii].resize({gi.r,gi.r});
              D2[ii].resize({gi.r,gi.r});
            }
            UUs.resize({gi.r,gi.r});
            VVs.resize({gi.r,gi.r});
            tmps.resize({gi.r,gi.r});
            sigma.resize({gi.r});

            Kad.update_shape({gi.dxx_mult,gi.r});
            efield.update_info(gi.r);
            compute_C.update_info(gi.r);
            compute_D.update_info(gi.r);
            gt::stop("Accept: decrease rank");

          }
          el_energyf << t << " " << el_energy << endl;
          t += tau;
          kk = kk + 1;
      } else{
        contf << "s" << endl;
        cout << "Accepted step, keeping same rank." << endl;
        gt::start("Accept keep rank: swap");
        lr_sol.X.swap(Xn);
        lr_sol.V.swap(Vn);
        lr_sol.S.swap(Sn);
        gt::stop("Accept keep rank: swap");

        el_energyf << t << " " << el_energy << endl;
        t += tau;
        kk = kk + 1;
      }

    }else if (ec == "ee"){
      gt::start("NEW el en (matmul, efield, ee)");
      blas.matmul(Xn,Sn,Kad);
      efield(Kad, Vn, Etmp, blas);

      double el_energy_new = electric_energy(Etmp, gi, blas);
      gt::stop("NEW el en (matmul, efield, ee)");
      gt::start("SVD decomposition");
      svd(Sn, UUs, VVs, sigma, blas);
      gt::stop("SVD decomposition");
      if(Sn.sl == stloc::host){
        sigma(gi.r-1) = 0.0;
      } else {
        #ifdef __CUDA__
          double ZERO = 0.0;
          cudaMemcpy(&sigma(gi.r-1),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
        #endif
      }

      gt::start("CUT el en (pmr, matmul, efield, ee)");
      //TODO: can be optimized, but it's r times r
      transpose_inplace(VVs);
      ptw_mult_row(VVs,sigma,tmps);
      blas.matmul(UUs,tmps,VVs);

      blas.matmul(Xn,VVs,Kad);
      efield(Kad, Vn, Etmp, blas);

      double el_energy_cut = electric_energy(Etmp, gi, blas);
      gt::stop("CUT el en (pmr, matmul, efield, ee)");

      double err_el_energy = abs(el_energy_new-el_energy_cut);

      if (err_el_energy >= (tol_inc+el_energy_new*tol_inc_rel)){
        if (gi.r == max_r){
          contf << "j" << endl;
          cout << "Should reject and increase rank but max rank reached. Proceeding keeping max rank." << endl;

          gt::start("Reject but max rank: swap");
          lr_sol.X.swap(Xn);
          lr_sol.V.swap(Vn);
          lr_sol.S.swap(Sn);
          gt::stop("Reject but max rank: swap");
 
          el_energyf << t << " " << el_energy << endl;
          t += tau;

          kk = kk + 1;
        } else {
          contf << "r" << endl;
          cout << "Rejected step, increasing rank by one." << endl;

          gt::start("Reject: increase rank");
          // Do all the updates
          gt::start("Reject: increase rank (updates)");
          
          lr_sol.S.swap(Sn);

          gi.update_rank(gi.r+1);
          lr_sol.update_info(gi.r);
          K_step_rk4.update_info(gi.r);
          S_step_rk4.update_info(gi.r);
          L_step_rk4.update_info(gi.r);

          Xn.update_shape({gi.dxx_mult,gi.r});
          Vn.update_shape({gi.dvv_mult,gi.r});
          gt::stop("Reject: increase rank (updates)");

          gt::start("Reject: increase rank (gram schmidt)");
          if(lr_sol.X.sl == stloc::host){
            mgs_orthcol_cpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);
            mgs_orthcol_cpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
          } else {
            #ifdef __CUDA__
              mgs_orthcol_gpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2],blas);
              mgs_orthcol_gpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2],blas);
            #endif
          }
          gt::stop("Reject: increase rank (gram schmidt)");

          gt::start("Reject: increase rank (some resizes rxr)");

          // This could directly become a kernel, but r times r so we'll see
          #ifdef __OPENMP__
          #pragma omp parallel for
          #endif
          for(Index i=0; i < lr_sol.S.num_elements(); i++){
              Index idx_r = i%gi.r;
              Index idx_c = i/gi.r;
              if((idx_r == (gi.r-1)) || (idx_c == (gi.r-1))){
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = 0.0;
                } else {
                  #ifdef __CUDA__
                    double ZERO = 0.0;
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
                  #endif
                }
              } else {
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                } else {
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
              }
          }
          Sn.resize({gi.r,gi.r});

          for(int ii = 0; ii < 3; ii++){
            C1[ii].resize({gi.r,gi.r});
            C2[ii].resize({gi.r,gi.r});
            D1[ii].resize({gi.r,gi.r});
            D2[ii].resize({gi.r,gi.r});
          }
          UUs.resize({gi.r,gi.r});
          VVs.resize({gi.r,gi.r});
          tmps.resize({gi.r,gi.r});
          sigma.resize({gi.r});

          Kad.update_shape({gi.dxx_mult,gi.r});
          efield.update_info(gi.r);
          compute_C.update_info(gi.r);
          compute_D.update_info(gi.r);
          gt::stop("Reject: increase rank (some resizes rxr)");
          gt::stop("Reject: increase rank");
        }
      } else if (err_el_energy <= (tol_dec+el_energy_new*tol_dec_rel)){
          if (gi.r == min_r){
            contf << "m" << endl;
            cout << "Accepted step, should decrease rank but min rank reached. Proceeding keeping min rank." << endl;

            gt::start("Accept but min rank: swap");
            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);
            lr_sol.S.swap(Sn);
            gt::stop("Accept but min rank: swap");

          } else {
            contf << "a" << endl;
            cout << "Accepted step, decreasing rank by one." << endl;

            gt::start("Accept: decrease rank");

            // Do all the updates
            gi.update_rank(gi.r-1);
            lr_sol.update_info(gi.r);
            K_step_rk4.update_info(gi.r);
            S_step_rk4.update_info(gi.r);
            L_step_rk4.update_info(gi.r);

            Xn.update_shape({gi.dxx_mult,gi.r});
            Vn.update_shape({gi.dvv_mult,gi.r});
            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);

            #ifdef __OPENMP__
            #pragma omp parallel for
            #endif
            for(Index i=0; i < lr_sol.S.num_elements(); i++){
                Index idx_r = i%gi.r;
                Index idx_c = i/gi.r;
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                }else{
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
            }
            Sn.resize({gi.r,gi.r});

            for(int ii = 0; ii < 3; ii++){
              C1[ii].resize({gi.r,gi.r});
              C2[ii].resize({gi.r,gi.r});
              D1[ii].resize({gi.r,gi.r});
              D2[ii].resize({gi.r,gi.r});
            }
            UUs.resize({gi.r,gi.r});
            VVs.resize({gi.r,gi.r});
            tmps.resize({gi.r,gi.r});
            sigma.resize({gi.r});

            Kad.update_shape({gi.dxx_mult,gi.r});
            efield.update_info(gi.r);
            compute_C.update_info(gi.r);
            compute_D.update_info(gi.r);
            gt::stop("Accept: decrease rank");

          }
          el_energyf << t << " " << el_energy << endl;
          t += tau;
          kk = kk + 1;
      } else{
        contf << "s" << endl;
        cout << "Accepted step, keeping same rank." << endl;
        gt::start("Accept keep rank: swap");
        lr_sol.X.swap(Xn);
        lr_sol.V.swap(Vn);
        lr_sol.S.swap(Sn);
        gt::stop("Accept keep rank: swap");

        el_energyf << t << " " << el_energy << endl;
        t += tau;
        kk = kk + 1;
      }

    } else{
      cout << "Error control not known" << endl;
      exit(1);
    }
    gt::stop("Rank choice");
  }
  gt::stop("Main loop");

  ofstream h_rank_f("h_rank_ps_"+ec+"_"+gstr+".data");
  for(Index i = 0; i < (Index)h_rank.size(); i++){
    h_rank_f << h_rank[i] << endl;
  }

  if (snapshots == 1){
    multi_array<double,2> K({gi.dxx_mult,gi.r},sl);
    multi_array<double,2> SOL({gi.dxx_mult,gi.dvv_mult},sl);
    blas.matmul(lr_sol.X, lr_sol.S, K);
    blas.matmul_transb(K, lr_sol.V, SOL);
    dump("solution_ps_"+ec+"_"+gstr+".data",SOL);
  }

}

void BUG_adapt(double final_time, double tau, int nsteps_int, grid_info_reserve<3>& gi, vector<const double*> X0, vector<const double*> V0, double tol, Index max_r, Index snapshots, const blas_ops& blas){

  gt::start("Initialization");

  if (2*gi.r > gi.rmax){
    cout << "ERROR: by construction in BUG the max rank must be at least twice the initial rank" << endl;
    exit(1);
  }

  stloc sl = (CPU) ? stloc::host : stloc::device;
  string gstr;

  orthogonalize gs(&blas);

  // Initialization
  lr2<double> lr_sol(gi.r,max_r,{gi.dxx_mult,gi.dvv_mult}, sl);

  if(sl == stloc::host) {
    initialize(lr_sol, X0, V0, gi.h_xx, gi.h_vv, blas);
    gstr = "cpu";
  } else {
    lr2<double> h_lr_sol(gi.r,max_r,{gi.dxx_mult,gi.dvv_mult}, stloc::host);
    initialize(h_lr_sol, X0, V0, gi.h_xx, gi.h_vv, blas);
    lr_sol = h_lr_sol;
    gstr = "gpu";
  }
  ofstream el_energyf("evolution_bug_"+gstr+".data");
  ofstream contf("contf_bug_"+gstr+".data");
  el_energyf.precision(16);
  double t = 0.0;
  Index n_steps = ceil(final_time/tau);

  vector<int> h_rank(n_steps);

  array<vec,3> E = create_vec_array(gi.dxx_mult, sl);
  array<vec,3> Etmp = create_vec_array(gi.dxx_mult, sl);

  PS_K_step_adapt K_step_rk4(sl, gi, &blas);
  PS_L_step_adapt L_step_rk4(sl, gi, &blas);
  BUG_S_step_adapt S_step_rk4(sl, gi, &blas, 2);

  mat Xn({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r}, sl);
  mat Vn({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r}, sl);

  ////
  mat Stmp({2*gi.r,2*gi.r}, sl);
  mat Mhat({2*gi.r,gi.r}, sl);
  mat Nhat({2*gi.r,gi.r}, sl);
  mat Shat({2*gi.r,2*gi.r}, sl);
  ////

  electric_field efield(sl, gi);

  coeff_C compute_C(sl, gi);
  coeff_D compute_D(sl, gi);

  array<mat, 3> C1 = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> C2 = create_mat_array({gi.r,gi.r}, sl);

  array<mat, 3> D1 = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> D2 = create_mat_array({gi.r,gi.r}, sl);

  // needed for error control
  mat UUs({2*gi.r,2*gi.r}, sl);
  mat VVs({2*gi.r,2*gi.r}, sl);
  vec sigma({2*gi.r}, sl);
  #ifdef __CUDA__
  vec sigma_h({2*gi.r}, stloc::host);
  #endif
  Index newr = gi.rmax/2;
  gt::stop("Initialization");

  cout.precision(8);
  gt::start("Main loop");
  for(Index kk=0; kk<n_steps; kk++){

    cout << "Step " << kk+1 << " of " << n_steps << endl;

    h_rank[kk] = (int)gi.r;

    gt::start("K step");
    // Compute K
    blas.matmul(lr_sol.X,lr_sol.S,Xn); // Xn is K
    gt::stop("K step");

    gt::start("Electric field");
    // Electric field
    efield(Xn, lr_sol.V, E, blas);
    gt::stop("Electric field");
    gt::start("Electric energy");
    double el_energy = electric_energy(E, gi, blas);
    gt::stop("Electric energy");

    cout << el_energy << endl;
    el_energyf << t << " " << el_energy << endl;

    // ---- K step ----
    gt::start("C coeffs");
    compute_C(lr_sol.V, C1, C2, blas);
    gt::stop("C coeffs");

    gt::start("K step");
    K_step_rk4(tau, Xn, E, C1, C2, nsteps_int);
    gt::stop("K step");


    // HERE AUGMENTATION K
    gt::start("K step aug + GS");
    Xn.update_shape({gi.dxx_mult,2*gi.r});
    if(Xn.sl == stloc::host){
      std::copy(lr_sol.X.data(), lr_sol.X.data()+lr_sol.X.num_elements(), Xn.data()+lr_sol.X.num_elements());
    } else {
      #ifdef __CUDA__
      cudaMemcpy(Xn.data()+lr_sol.X.num_elements(), lr_sol.X.data(), sizeof(double)*lr_sol.X.num_elements(),cudaMemcpyDeviceToDevice);
      #endif
    }
    gs(Xn, Stmp, gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);
    gt::stop("K step aug + GS");

    // ---- L step ----
    gt::start("D coeffs");
    compute_D(lr_sol.X, E, D1, D2, blas);
    gt::stop("D coeffs");

    gt::start("L step");
    blas.matmul_transb(lr_sol.V,lr_sol.S,Vn); // Vn is L
    L_step_rk4(tau, Vn, D1, D2, nsteps_int);
    gt::stop("L step");

    // HERE AUGMENTATION L
    gt::start("L step aug + GS");
    Vn.update_shape({gi.dvv_mult,2*gi.r});
    if(Vn.sl == stloc::host){
      std::copy(lr_sol.V.data(), lr_sol.V.data()+lr_sol.V.num_elements(), Vn.data()+lr_sol.V.num_elements());
    } else {
      #ifdef __CUDA__
      cudaMemcpy(Vn.data()+lr_sol.V.num_elements(), lr_sol.V.data(), sizeof(double)*lr_sol.V.num_elements(),cudaMemcpyDeviceToDevice);
      #endif
    }
    gs(Vn, Stmp, gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
    gt::stop("L step aug + GS");

    // ---- S step ----
    gt::start("Hat matrices (for S step)");
    blas.matmul_transa(Xn,lr_sol.X,Mhat);
    Mhat *= gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2];
    blas.matmul_transa(Vn,lr_sol.V,Nhat);
    Nhat *= gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2];

    Stmp.resize({2*gi.r,gi.r});
    blas.matmul(Mhat,lr_sol.S,Stmp);
    blas.matmul_transb(Stmp,Nhat,Shat);
    gt::stop("Hat matrices (for S step)");

    gt::start("New C and D coeffs (for S step)");
    // Recompute C and D coeffs, fixed electric field
    for(int ii = 0; ii < 3; ii++){
      C1[ii].resize({2*gi.r,2*gi.r});
      C2[ii].resize({2*gi.r,2*gi.r});
      D1[ii].resize({2*gi.r,2*gi.r});
      D2[ii].resize({2*gi.r,2*gi.r});
    }
    compute_C.update_info(2*gi.r);
    compute_D.update_info(2*gi.r);
    compute_C(Vn, C1, C2, blas);
    compute_D(Xn, E, D1, D2, blas);
    gt::stop("New C and D coeffs (for S step)");

    gt::start("S step");
    S_step_rk4(tau, Shat, C1, C2, D1, D2, nsteps_int);
    gt::stop("S step");

    // Determine new rank
    gt::start("SVD decomposition");
    svd(Shat, UUs, VVs, sigma, blas);
    gt::stop("SVD decomposition");

    gt::start("Choosing new rank");
    if(Shat.sl == stloc::host){
      for(Index idx = 0; idx < 2*gi.r; idx++){
        double psum = 0;
        for(Index ii = idx; ii < 2*gi.r; ii++){
          psum += sigma(ii)*sigma(ii);
        }
        if (sqrt(psum) <= tol){
          newr = idx+1;
          break;
        }
      }
    } else {
      #ifdef __CUDA__
      cudaMemcpy(sigma_h.data(), sigma.data(), sizeof(double)*(2*gi.r),cudaMemcpyDeviceToHost);

      for(Index idx = 0; idx < 2*gi.r; idx++){
        double psum = 0;
        for(Index ii = idx; ii < 2*gi.r; ii++){
          psum += sigma_h(ii)*sigma_h(ii);
        }
        if (sqrt(psum) <= tol){
          newr = idx+1;
          break;
        }
      }
      #endif
    }

    if (2*newr > gi.rmax){
      cout << "Rank can't exceed maxrank, setting rank=rmax/2" << endl;
      newr = gi.rmax/2;
    }
    gt::stop("Choosing new rank");

    gt::start("Truncate and updates");
    // Now truncate and do all the updates
    gi.update_rank(newr);
    lr_sol.update_info(gi.r);
    if (lr_sol.S.sl == stloc::host){
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0; i < lr_sol.S.num_elements(); i++){
        Index idx_r = i%gi.r;
        Index idx_c = i/gi.r;
        if(idx_r == idx_c){
          lr_sol.S(idx_r,idx_c) = sigma(idx_r);
        } else {
          lr_sol.S(idx_r,idx_c) = 0.0;
        }
      }
    } else {
      #ifdef __CUDA__
      diag_S<<<gi.r,gi.r>>>(gi.r,lr_sol.S.data(),sigma.data());
      cudaDeviceSynchronize();
      #endif
    }
    UUs.update_shape({2*gi.r,gi.r});
    VVs.update_shape({2*gi.r,gi.r});
    blas.matmul(Xn,UUs,lr_sol.X);
    blas.matmul(Vn,VVs,lr_sol.V);

    K_step_rk4.update_info(gi.r);
    S_step_rk4.update_info(gi.r);
    L_step_rk4.update_info(gi.r);
    Xn.update_shape({gi.dxx_mult,gi.r});
    Vn.update_shape({gi.dvv_mult,gi.r});

    Stmp.resize({2*gi.r,2*gi.r});
    Mhat.resize({2*gi.r,gi.r});
    Nhat.resize({2*gi.r,gi.r});
    Shat.resize({2*gi.r,2*gi.r});

    efield.update_info(gi.r);
    compute_C.update_info(gi.r);
    compute_D.update_info(gi.r);
    for(int ii = 0; ii < 3; ii++){
      C1[ii].resize({gi.r,gi.r});
      C2[ii].resize({gi.r,gi.r});
      D1[ii].resize({gi.r,gi.r});
      D2[ii].resize({gi.r,gi.r});
    }
    UUs.resize({2*gi.r,2*gi.r});
    VVs.resize({2*gi.r,2*gi.r});
    sigma.resize({2*gi.r});
    #ifdef __CUDA__
    sigma_h.resize({2*gi.r});
    #endif
    gt::stop("Truncate and updates");

    t += tau;

  }
  gt::stop("Main loop");

  ofstream h_rank_f("h_rank_bug_"+gstr+".data");
  for(Index i = 0; i < (Index)h_rank.size(); i++){
    h_rank_f << h_rank[i] << endl;
  }
  
  if (snapshots == 1){
    multi_array<double,2> K({gi.dxx_mult,gi.r},sl);
    multi_array<double,2> SOL({gi.dxx_mult,gi.dvv_mult},sl);
    blas.matmul(lr_sol.X, lr_sol.S, K);
    blas.matmul_transb(K, lr_sol.V, SOL);
    dump("solution_bug_"+gstr+".data",SOL);
  }
}

void PS_so_adapt(double final_time, double tau, int nsteps_int, grid_info_reserve<3>& gi, vector<const double*> X0, vector<const double*> V0, double tol_inc, double tol_inc_rel, double tol_dec, double tol_dec_rel, Index min_r, Index max_r, string ec, Index snapshots, const blas_ops& blas){

  stloc sl = (CPU) ? stloc::host : stloc::device;
  string gstr;

  orthogonalize gs(&blas);

  // Initialization
  lr2<double> lr_sol(gi.r,max_r,{gi.dxx_mult,gi.dvv_mult}, sl);

  if(sl == stloc::host) {
    initialize(lr_sol, X0, V0, gi.h_xx, gi.h_vv, blas);
    gstr = "cpu";
  } else {
    lr2<double> h_lr_sol(gi.r,max_r,{gi.dxx_mult,gi.dvv_mult}, stloc::host);
    initialize(h_lr_sol, X0, V0, gi.h_xx, gi.h_vv, blas);
    lr_sol = h_lr_sol;
    gstr = "gpu";
  }
  ofstream el_energyf("evolution_ps_so_"+ec+"_"+gstr+".data");
  ofstream contf("contf_ps_so_"+ec+"_"+gstr+".data");
  el_energyf.precision(16);
  double t = 0.0;
  Index n_steps = ceil(final_time/tau);

  vector<int> h_rank(n_steps);

  array<vec,3> E = create_vec_array(gi.dxx_mult, sl);
  array<vec,3> Etmp = create_vec_array(gi.dxx_mult, sl);
  array<vec,3> Etmp2 = create_vec_array(gi.dxx_mult, sl);

  PS_K_step_adapt K_step_rk4(sl, gi, &blas);
  PS_S_step_adapt S_step_rk4(sl, gi, &blas);
  PS_L_step_adapt L_step_rk4(sl, gi, &blas);

  mat Xn({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r}, sl);
  mat XnT({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r}, sl);
  mat Sn({gi.r,gi.r}, sl);
  mat Vn({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r}, sl);

  electric_field efield(sl, gi);

  coeff_C compute_C(sl, gi);
  coeff_D compute_D(sl, gi);

  array<mat, 3> C1 = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> C2 = create_mat_array({gi.r,gi.r}, sl);

  array<mat, 3> D1 = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> D2 = create_mat_array({gi.r,gi.r}, sl);

  // needed for error control
  mat Kad({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r}, sl);
  mat UUs({gi.r,gi.r}, sl);
  mat VVs({gi.r,gi.r}, sl);
  vec sigma({gi.r}, sl);
  mat tmps({gi.r,gi.r}, sl);

  Index kk = 1;
  cout.precision(8);
  while(kk<=n_steps){

    cout << "Step " << kk << " of " << n_steps << endl;

    h_rank[kk-1] = (int)gi.r;

    // Compute electric field at time tau/2
    
    // Compute C coefficients
    compute_C(lr_sol.V, C1, C2, blas);
    // Compute K
    blas.matmul(lr_sol.X,lr_sol.S,Xn); // Xn is K
    XnT = Xn;
    // Electric field and energy
    efield(Xn, lr_sol.V, E, blas);
    double el_energy = electric_energy(E, gi, blas);
    cout << el_energy << endl;
    // ---- K step ----
    K_step_rk4(tau/2.0, Xn, E, C1, C2, nsteps_int);
    gs(Xn, Sn, gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]); // Xn the new X
    // Compute D coefficients
    compute_D(Xn, E, D1, D2, blas);
    // ---- S step ----
    S_step_rk4(tau/2.0, Sn, C1, C2, D1, D2, nsteps_int);
    // Compute L
    blas.matmul_transb(lr_sol.V,Sn,Vn); // Vn is L
    // ---- L step ----
    L_step_rk4(tau/2.0, Vn, D1, D2, nsteps_int);
    // Electric field at tau/2
    efield(Xn, Vn, E, blas);
    
    // Restart integration using Strang splitting
    // ---- Half step K step ----
    K_step_rk4(tau/2.0, XnT, E, C1, C2, nsteps_int);
    gs(XnT, Sn, gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]); // Xn the new X
    // Compute D coefficients
    compute_D(XnT, E, D1, D2, blas);
    // ---- Half step S step ----
    S_step_rk4(tau/2.0, Sn, C1, C2, D1, D2, nsteps_int);
    // Compute L
    blas.matmul_transb(lr_sol.V,Sn,Vn); // Vn is L
    // ---- Full step L step ----
    L_step_rk4(tau, Vn, D1, D2, nsteps_int);
    gs(Vn, Sn, gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
    transpose_inplace(Sn);
    // Compute C coefficients
    compute_C(Vn, C1, C2, blas);
    // ---- Half step S step ----
    S_step_rk4(tau/2.0, Sn, C1, C2, D1, D2, nsteps_int);
    // Compute K
    blas.matmul(XnT,Sn,Xn); // Xn is K
    // ---- Half step K step ----
    K_step_rk4(tau/2.0, Xn, E, C1, C2, nsteps_int);
    gs(Xn, Sn, gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]); // Xn the new X

    if (ec == "f"){
      svd(Sn, UUs, VVs, sigma, blas);
      double svr=0;
      if(Sn.sl == stloc::host){
        svr = sigma(gi.r-1);
      } else {
        #ifdef __CUDA__
          cudaMemcpy(&svr, &sigma(gi.r-1), sizeof(double),cudaMemcpyDeviceToHost);
        #endif
      }

      if (svr >= tol_inc){
        if (gi.r == max_r){
          contf << "j" << endl;
          cout << "Should reject and increase rank but max rank reached. Proceeding keeping max rank." << endl;

          lr_sol.X.swap(Xn);
          lr_sol.V.swap(Vn);
          lr_sol.S.swap(Sn);
 
          el_energyf << t << " " << el_energy << endl;
          t += tau;

          kk = kk + 1;
        } else {
          contf << "r" << endl;
          cout << "Rejected step, increasing rank by one." << endl;

          // Do all the updates

          lr_sol.S.swap(Sn);

          gi.update_rank(gi.r+1);
          lr_sol.update_info(gi.r);
          K_step_rk4.update_info(gi.r);
          S_step_rk4.update_info(gi.r);
          L_step_rk4.update_info(gi.r);

          Xn.update_shape({gi.dxx_mult,gi.r});
          XnT.update_shape({gi.dxx_mult,gi.r});
          Vn.update_shape({gi.dvv_mult,gi.r});

          if(lr_sol.X.sl == stloc::host){
            mgs_orthcol_cpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);
            mgs_orthcol_cpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
          } else {
            #ifdef __CUDA__
              mgs_orthcol_gpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2],blas);
              mgs_orthcol_gpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2],blas);
            #endif
          }


          // This could directly become a kernel, but s times s so we'll see
          #ifdef __OPENMP__
          #pragma omp parallel for
          #endif
          for(Index i=0; i < lr_sol.S.num_elements(); i++){
              Index idx_r = i%gi.r;
              Index idx_c = i/gi.r;
              if((idx_r == (gi.r-1)) || (idx_c == (gi.r-1))){
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = 0.0;
                } else {
                  #ifdef __CUDA__
                    double ZERO = 0.0;
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
                  #endif
                }
              }
              else {
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                } else {
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
              }
          }
          Sn.resize({gi.r,gi.r});
          
          for(int ii = 0; ii < 3; ii++){
            C1[ii].resize({gi.r,gi.r});
            C2[ii].resize({gi.r,gi.r});
            D1[ii].resize({gi.r,gi.r});
            D2[ii].resize({gi.r,gi.r});
          }
          UUs.resize({gi.r,gi.r});
          VVs.resize({gi.r,gi.r});
          tmps.resize({gi.r,gi.r});
          sigma.resize({gi.r});

          Kad.update_shape({gi.dxx_mult,gi.r});
          efield.update_info(gi.r);
          compute_C.update_info(gi.r);
          compute_D.update_info(gi.r);
        }
      } else if (svr <= tol_dec){
          if (gi.r == min_r){
            contf << "m" << endl;
            cout << "Accepted step, should decrease rank but min rank reached. Proceeding keeping min rank." << endl;

            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);
            lr_sol.S.swap(Sn);


          } else {
            contf << "a" << endl;
            cout << "Accepted step, decreasing rank by one." << endl;


            // Do all the updates
            gi.update_rank(gi.r-1);
            lr_sol.update_info(gi.r);
            K_step_rk4.update_info(gi.r);
            S_step_rk4.update_info(gi.r);
            L_step_rk4.update_info(gi.r);

            Xn.update_shape({gi.dxx_mult,gi.r});
            XnT.update_shape({gi.dxx_mult,gi.r});
            Vn.update_shape({gi.dvv_mult,gi.r});
            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);

            #ifdef __OPENMP__
            #pragma omp parallel for
            #endif
            for(Index i=0; i < lr_sol.S.num_elements(); i++){
                Index idx_r = i%gi.r;
                Index idx_c = i/gi.r;
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                }else{
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
            }
            Sn.resize({gi.r,gi.r});

            for(int ii = 0; ii < 3; ii++){
              C1[ii].resize({gi.r,gi.r});
              C2[ii].resize({gi.r,gi.r});
              D1[ii].resize({gi.r,gi.r});
              D2[ii].resize({gi.r,gi.r});
            }
            UUs.resize({gi.r,gi.r});
            VVs.resize({gi.r,gi.r});
            tmps.resize({gi.r,gi.r});
            sigma.resize({gi.r});

            Kad.update_shape({gi.dxx_mult,gi.r});
            efield.update_info(gi.r);
            compute_C.update_info(gi.r);
            compute_D.update_info(gi.r);

          }
          el_energyf << t << " " << el_energy << endl;
          t += tau;
          kk = kk + 1;
      } else{
        contf << "s" << endl;
        cout << "Accepted step, keeping same rank." << endl;
        lr_sol.X.swap(Xn);
        lr_sol.V.swap(Vn);
        lr_sol.S.swap(Sn);

        el_energyf << t << " " << el_energy << endl;
        t += tau;
        kk = kk + 1;
      }
    }else if (ec == "ef"){
      blas.matmul(Xn,Sn,Kad);
      efield(Kad, Vn, Etmp, blas);

      svd(Sn, UUs, VVs, sigma, blas);
      if(Sn.sl == stloc::host){
        sigma(gi.r-1) = 0.0;
      } else {
        #ifdef __CUDA__
          double ZERO = 0.0;
          cudaMemcpy(&sigma(gi.r-1),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
        #endif
      }

      //TODO: can be optimized, but it's r times r
      transpose_inplace(VVs);
      ptw_mult_row(VVs,sigma,tmps);
      blas.matmul(UUs,tmps,VVs);

      blas.matmul(Xn,VVs,Kad);
      efield(Kad, Vn, Etmp2, blas);

      double norm_ef = norm_EF(Etmp, gi, blas);
      double err_ef = norm_err_EF(Etmp, Etmp2, gi, blas);

      if (err_ef >= (tol_inc+norm_ef*tol_inc_rel)){
        if (gi.r == max_r){
          contf << "j" << endl;
          cout << "Should reject and increase rank but max rank reached. Proceeding keeping max rank." << endl;

          lr_sol.X.swap(Xn);
          lr_sol.V.swap(Vn);
          lr_sol.S.swap(Sn);
 
          el_energyf << t << " " << el_energy << endl;
          t += tau;

          kk = kk + 1;
        } else {
          contf << "r" << endl;
          cout << "Rejected step, increasing rank by one." << endl;

          // Do all the updates
          
          lr_sol.S.swap(Sn);

          gi.update_rank(gi.r+1);
          lr_sol.update_info(gi.r);
          K_step_rk4.update_info(gi.r);
          S_step_rk4.update_info(gi.r);
          L_step_rk4.update_info(gi.r);

          Xn.update_shape({gi.dxx_mult,gi.r});
          XnT.update_shape({gi.dxx_mult,gi.r});
          Vn.update_shape({gi.dvv_mult,gi.r});

          if(lr_sol.X.sl == stloc::host){
            mgs_orthcol_cpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);
            mgs_orthcol_cpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
          } else {
            #ifdef __CUDA__
              mgs_orthcol_gpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2],blas);
              mgs_orthcol_gpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2],blas);
            #endif
          }


          // This could directly become a kernel, but r times r so we'll see
          #ifdef __OPENMP__
          #pragma omp parallel for
          #endif
          for(Index i=0; i < lr_sol.S.num_elements(); i++){
              Index idx_r = i%gi.r;
              Index idx_c = i/gi.r;
              if((idx_r == (gi.r-1)) || (idx_c == (gi.r-1))){
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = 0.0;
                } else {
                  #ifdef __CUDA__
                    double ZERO = 0.0;
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
                  #endif
                }
              } else {
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                } else {
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
              }
          }
          Sn.resize({gi.r,gi.r});

          for(int ii = 0; ii < 3; ii++){
            C1[ii].resize({gi.r,gi.r});
            C2[ii].resize({gi.r,gi.r});
            D1[ii].resize({gi.r,gi.r});
            D2[ii].resize({gi.r,gi.r});
          }
          UUs.resize({gi.r,gi.r});
          VVs.resize({gi.r,gi.r});
          tmps.resize({gi.r,gi.r});
          sigma.resize({gi.r});

          Kad.update_shape({gi.dxx_mult,gi.r});
          efield.update_info(gi.r);
          compute_C.update_info(gi.r);
          compute_D.update_info(gi.r);
        }
      } else if (err_ef <= (tol_dec+norm_ef*tol_dec_rel)){
          if (gi.r == min_r){
            contf << "m" << endl;
            cout << "Accepted step, should decrease rank but min rank reached. Proceeding keeping min rank." << endl;

            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);
            lr_sol.S.swap(Sn);

          } else {
            contf << "a" << endl;
            cout << "Accepted step, decreasing rank by one." << endl;

            // Do all the updates
            gi.update_rank(gi.r-1);
            lr_sol.update_info(gi.r);
            K_step_rk4.update_info(gi.r);
            S_step_rk4.update_info(gi.r);
            L_step_rk4.update_info(gi.r);

            Xn.update_shape({gi.dxx_mult,gi.r});
            XnT.update_shape({gi.dxx_mult,gi.r});
            Vn.update_shape({gi.dvv_mult,gi.r});
            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);

            #ifdef __OPENMP__
            #pragma omp parallel for
            #endif
            for(Index i=0; i < lr_sol.S.num_elements(); i++){
                Index idx_r = i%gi.r;
                Index idx_c = i/gi.r;
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                }else{
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
            }
            Sn.resize({gi.r,gi.r});

            for(int ii = 0; ii < 3; ii++){
              C1[ii].resize({gi.r,gi.r});
              C2[ii].resize({gi.r,gi.r});
              D1[ii].resize({gi.r,gi.r});
              D2[ii].resize({gi.r,gi.r});
            }
            UUs.resize({gi.r,gi.r});
            VVs.resize({gi.r,gi.r});
            tmps.resize({gi.r,gi.r});
            sigma.resize({gi.r});

            Kad.update_shape({gi.dxx_mult,gi.r});
            efield.update_info(gi.r);
            compute_C.update_info(gi.r);
            compute_D.update_info(gi.r);

          }
          el_energyf << t << " " << el_energy << endl;
          t += tau;
          kk = kk + 1;
      } else{
        contf << "s" << endl;
        cout << "Accepted step, keeping same rank." << endl;
        lr_sol.X.swap(Xn);
        lr_sol.V.swap(Vn);
        lr_sol.S.swap(Sn);

        el_energyf << t << " " << el_energy << endl;
        t += tau;
        kk = kk + 1;
      }

    }else if (ec == "ee"){
      blas.matmul(Xn,Sn,Kad);
      efield(Kad, Vn, Etmp, blas);

      double el_energy_new = electric_energy(Etmp, gi, blas);
      svd(Sn, UUs, VVs, sigma, blas);
      if(Sn.sl == stloc::host){
        sigma(gi.r-1) = 0.0;
      } else {
        #ifdef __CUDA__
          double ZERO = 0.0;
          cudaMemcpy(&sigma(gi.r-1),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
        #endif
      }

      //TODO: can be optimized, but it's r times r
      transpose_inplace(VVs);
      ptw_mult_row(VVs,sigma,tmps);
      blas.matmul(UUs,tmps,VVs);

      blas.matmul(Xn,VVs,Kad);
      efield(Kad, Vn, Etmp, blas);

      double el_energy_cut = electric_energy(Etmp, gi, blas);

      double err_el_energy = abs(el_energy_new-el_energy_cut);

      if (err_el_energy >= (tol_inc+el_energy_new*tol_inc_rel)){
        if (gi.r == max_r){
          contf << "j" << endl;
          cout << "Should reject and increase rank but max rank reached. Proceeding keeping max rank." << endl;

          lr_sol.X.swap(Xn);
          lr_sol.V.swap(Vn);
          lr_sol.S.swap(Sn);
 
          el_energyf << t << " " << el_energy << endl;
          t += tau;

          kk = kk + 1;
        } else {
          contf << "r" << endl;
          cout << "Rejected step, increasing rank by one." << endl;

          // Do all the updates
          
          lr_sol.S.swap(Sn);

          gi.update_rank(gi.r+1);
          lr_sol.update_info(gi.r);
          K_step_rk4.update_info(gi.r);
          S_step_rk4.update_info(gi.r);
          L_step_rk4.update_info(gi.r);

          Xn.update_shape({gi.dxx_mult,gi.r});
          XnT.update_shape({gi.dxx_mult,gi.r});
          Vn.update_shape({gi.dvv_mult,gi.r});

          if(lr_sol.X.sl == stloc::host){
            mgs_orthcol_cpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);
            mgs_orthcol_cpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
          } else {
            #ifdef __CUDA__
              mgs_orthcol_gpu(lr_sol.X,gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2],blas);
              mgs_orthcol_gpu(lr_sol.V,gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2],blas);
            #endif
          }


          // This could directly become a kernel, but r times r so we'll see
          #ifdef __OPENMP__
          #pragma omp parallel for
          #endif
          for(Index i=0; i < lr_sol.S.num_elements(); i++){
              Index idx_r = i%gi.r;
              Index idx_c = i/gi.r;
              if((idx_r == (gi.r-1)) || (idx_c == (gi.r-1))){
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = 0.0;
                } else {
                  #ifdef __CUDA__
                    double ZERO = 0.0;
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&ZERO, sizeof(double),cudaMemcpyHostToDevice);
                  #endif
                }
              } else {
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                } else {
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
              }
          }
          Sn.resize({gi.r,gi.r});

          for(int ii = 0; ii < 3; ii++){
            C1[ii].resize({gi.r,gi.r});
            C2[ii].resize({gi.r,gi.r});
            D1[ii].resize({gi.r,gi.r});
            D2[ii].resize({gi.r,gi.r});
          }
          UUs.resize({gi.r,gi.r});
          VVs.resize({gi.r,gi.r});
          tmps.resize({gi.r,gi.r});
          sigma.resize({gi.r});

          Kad.update_shape({gi.dxx_mult,gi.r});
          efield.update_info(gi.r);
          compute_C.update_info(gi.r);
          compute_D.update_info(gi.r);
        }
      } else if (err_el_energy <= (tol_dec+el_energy_new*tol_dec_rel)){
          if (gi.r == min_r){
            contf << "m" << endl;
            cout << "Accepted step, should decrease rank but min rank reached. Proceeding keeping min rank." << endl;

            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);
            lr_sol.S.swap(Sn);

          } else {
            contf << "a" << endl;
            cout << "Accepted step, decreasing rank by one." << endl;


            // Do all the updates
            gi.update_rank(gi.r-1);
            lr_sol.update_info(gi.r);
            K_step_rk4.update_info(gi.r);
            S_step_rk4.update_info(gi.r);
            L_step_rk4.update_info(gi.r);

            Xn.update_shape({gi.dxx_mult,gi.r});
            XnT.update_shape({gi.dxx_mult,gi.r});
            Vn.update_shape({gi.dvv_mult,gi.r});
            lr_sol.X.swap(Xn);
            lr_sol.V.swap(Vn);

            #ifdef __OPENMP__
            #pragma omp parallel for
            #endif
            for(Index i=0; i < lr_sol.S.num_elements(); i++){
                Index idx_r = i%gi.r;
                Index idx_c = i/gi.r;
                if(lr_sol.S.sl == stloc::host){
                  lr_sol.S(idx_r,idx_c) = Sn(idx_r,idx_c);
                }else{
                  #ifdef __CUDA__
                    cudaMemcpy(&lr_sol.S(idx_r,idx_c),&Sn(idx_r,idx_c), sizeof(double),cudaMemcpyDeviceToDevice);
                  #endif
                }
            }
            Sn.resize({gi.r,gi.r});

            for(int ii = 0; ii < 3; ii++){
              C1[ii].resize({gi.r,gi.r});
              C2[ii].resize({gi.r,gi.r});
              D1[ii].resize({gi.r,gi.r});
              D2[ii].resize({gi.r,gi.r});
            }
            UUs.resize({gi.r,gi.r});
            VVs.resize({gi.r,gi.r});
            tmps.resize({gi.r,gi.r});
            sigma.resize({gi.r});

            Kad.update_shape({gi.dxx_mult,gi.r});
            efield.update_info(gi.r);
            compute_C.update_info(gi.r);
            compute_D.update_info(gi.r);

          }
          el_energyf << t << " " << el_energy << endl;
          t += tau;
          kk = kk + 1;
      } else{
        contf << "s" << endl;
        cout << "Accepted step, keeping same rank." << endl;
        lr_sol.X.swap(Xn);
        lr_sol.V.swap(Vn);
        lr_sol.S.swap(Sn);

        el_energyf << t << " " << el_energy << endl;
        t += tau;
        kk = kk + 1;
      }

    } else{
      cout << "Error control not known" << endl;
      exit(1);
    }
  }

  ofstream h_rank_f("h_rank_ps_so_"+ec+"_"+gstr+".data");
  for(Index i = 0; i < (Index)h_rank.size(); i++){
    h_rank_f << h_rank[i] << endl;
  }


  if (snapshots == 1){
    //multi_array<double,2> K({gi.dxx_mult,gi.r},sl);
    //multi_array<double,2> SOL({gi.dxx_mult,gi.dvv_mult},sl);
    //blas.matmul(lr_sol.X, lr_sol.S, K);
    //blas.matmul_transb(K, lr_sol.V, SOL);
    //dump("solution_ps_so_"+ec+"_"+gstr+".data",SOL);

    mat TMPX({gi.N_xx[0],gi.r}, sl);
    mat TMPV({gi.N_vv[0],gi.r}, sl);
    mat TMP1({gi.N_xx[0],gi.r}, sl);
    mat SOL({gi.N_xx[0],gi.N_vv[0]}, sl);

    extract_rows(lr_sol.X, 0, gi.N_xx[0]-1, TMPX);
    extract_rows(lr_sol.V, pow(gi.N_vv[0],3)/2+pow(gi.N_vv[0],2)/2, pow(gi.N_vv[0],3)/2+pow(gi.N_vv[0],2)/2+gi.N_vv[0]-1, TMPV);
    blas.matmul(TMPX, lr_sol.S, TMP1);
    blas.matmul_transb(TMP1, TMPV, SOL);
    dump("slice_ps_so_"+gstr+".data",SOL);

    mat EX({gi.N_xx[0],1}, sl);
    mat EY({gi.N_xx[0],1}, sl);
    mat EZ({gi.N_xx[0],1}, sl);
    if (TMPX.sl == stloc::host){
      std::copy(E[0].data(),E[0].data()+gi.N_xx[0]+1,&EX({0,0}));
      std::copy(E[1].data(),E[1].data()+gi.N_xx[0]+1,&EY({0,0}));
      std::copy(E[2].data(),E[2].data()+gi.N_xx[0]+1,&EZ({0,0}));
    } else{
      #ifdef __CUDA__
      cudaMemcpy(&EX({0,0}), E[0].data(), sizeof(double)*(gi.N_xx[0]),cudaMemcpyDeviceToDevice);
      cudaMemcpy(&EY({0,0}), E[1].data(), sizeof(double)*(gi.N_xx[0]),cudaMemcpyDeviceToDevice);
      cudaMemcpy(&EZ({0,0}), E[2].data(), sizeof(double)*(gi.N_xx[0]),cudaMemcpyDeviceToDevice);
      #endif
    }

    dump("EX_"+ec+"_"+gstr+".data",EX);
    dump("EY_"+ec+"_"+gstr+".data",EY);
    dump("EZ_"+ec+"_"+gstr+".data",EZ);


  }

}

void BUG_so_adapt(double final_time, double tau, int nsteps_int, grid_info_reserve<3>& gi, vector<const double*> X0, vector<const double*> V0, double tol, Index max_r, Index snapshots, const blas_ops& blas){

  // MIDPOINT BUG WITH 3*r AUGMENTED BASIS

  if (3*gi.r > gi.rmax){
    cout << "ERROR: by construction in BUG second order the max rank must be at least three times the initial rank" << endl;
    exit(1);
  }

  stloc sl = (CPU) ? stloc::host : stloc::device;
  string gstr;

  orthogonalize gs(&blas);

  // Initialization
  lr2<double> lr_sol(gi.r,max_r,{gi.dxx_mult,gi.dvv_mult}, sl);

  if(sl == stloc::host) {
    initialize(lr_sol, X0, V0, gi.h_xx, gi.h_vv, blas);
    gstr = "cpu";
  } else {
    lr2<double> h_lr_sol(gi.r,max_r,{gi.dxx_mult,gi.dvv_mult}, stloc::host);
    initialize(h_lr_sol, X0, V0, gi.h_xx, gi.h_vv, blas);
    lr_sol = h_lr_sol;
    gstr = "gpu";
  }
  ofstream el_energyf("evolution_bug_so_"+gstr+".data");
  ofstream contf("contf_bug_so_"+gstr+".data");
  el_energyf.precision(16);
  double t = 0.0;
  Index n_steps = ceil(final_time/tau);

  vector<int> h_rank(n_steps);

  array<vec,3> E = create_vec_array(gi.dxx_mult, sl);

  PS_K_step_adapt K_step_rk4(sl, gi, &blas);
  PS_L_step_adapt L_step_rk4(sl, gi, &blas);
  BUG_S_step_adapt S_step_rk4(sl, gi, &blas, 3);

  BUG_S_step_adapt S_step_rk4_BUG1(sl, gi, &blas, 1);

  mat Xn({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r}, sl);
  mat XnT({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r}, sl);
  mat FnX({gi.dxx_mult,gi.rmax},{gi.dxx_mult,gi.r}, sl);
  mat Vn({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r}, sl);
  mat VnT({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r}, sl);
  mat FnV({gi.dvv_mult,gi.rmax},{gi.dvv_mult,gi.r}, sl);

  array<vec,3> v = create_vec_array(gi.dvv_mult,sl);
  v = L_step_rk4.get_v();

  ////
  mat Sn({gi.r,gi.r}, sl);
  mat M({gi.r,gi.r}, sl);
  mat N({gi.r,gi.r}, sl);
  mat Stmp({gi.r,gi.r}, sl);

  mat Mbar({3*gi.r,gi.r}, sl);
  mat Nbar({3*gi.r,gi.r}, sl);
  mat Sbar({3*gi.r,3*gi.r}, sl);
  ////

  electric_field efield(sl, gi);

  coeff_C compute_C(sl, gi);
  coeff_D compute_D(sl, gi);

  array<mat, 3> C1 = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> C2 = create_mat_array({gi.r,gi.r}, sl);

  array<mat, 3> D1 = create_mat_array({gi.r,gi.r}, sl);
  array<mat, 3> D2 = create_mat_array({gi.r,gi.r}, sl);

  // needed for error control
  mat UUs({3*gi.r,3*gi.r}, sl);
  mat VVs({3*gi.r,3*gi.r}, sl);
  vec sigma({3*gi.r}, sl);
  #ifdef __CUDA__
  vec sigma_h({3*gi.r}, stloc::host);
  #endif
  Index newr = gi.rmax/3;

  cout.precision(8);
  for(Index kk=0; kk<n_steps; kk++){

    cout << "Step " << kk+1 << " of " << n_steps << endl;

    h_rank[kk] = (int)gi.r;

    // --- HALF STEP FIXED-RANK BUG ---

    // Compute K
    blas.matmul(lr_sol.X,lr_sol.S,Xn); // Xn is K

    // Electric field
    efield(Xn, lr_sol.V, E, blas);
    double el_energy = electric_energy(E, gi, blas);
    cout << el_energy << endl;
    el_energyf << t << " " << el_energy << endl;

    // K step
    compute_C(lr_sol.V, C1, C2, blas);
    K_step_rk4(tau/2.0, Xn, E, C1, C2, nsteps_int);
    gs(Xn, M, gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);

    // L step
    compute_D(lr_sol.X, E, D1, D2, blas);

    blas.matmul_transb(lr_sol.V,lr_sol.S,Vn); // Vn is L
    L_step_rk4(tau/2.0, Vn, D1, D2, nsteps_int);
    gs(Vn, N, gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);
 
    // S step
    blas.matmul_transa(Xn,lr_sol.X,M);
    M *= gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2];

    blas.matmul_transa(Vn,lr_sol.V,N);
    N *= gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2];

    blas.matmul(M,lr_sol.S,Stmp);
    blas.matmul_transb(Stmp,N,Sn);

    compute_C(Vn, C1, C2, blas);
    compute_D(Xn, E, D1, D2, blas);

    S_step_rk4_BUG1(tau/2.0, Sn, C1, C2, D1, D2, nsteps_int);

    // --- Electric field at half step ---
    blas.matmul(Xn,Sn,XnT);
    efield(XnT, Vn, E, blas);

    // --- Do basis augmentation ---
    compute_D(Xn, E, D1, D2, blas);

    // For X
    K_step_rk4.rk4_K_rhs(XnT,E,C1,C2,FnX);
    FnX *= tau;

    Xn.update_shape({gi.dxx_mult,3*gi.r});
    if(Xn.sl == stloc::host){
      std::copy(lr_sol.X.data(), lr_sol.X.data()+lr_sol.X.num_elements(), Xn.data()+lr_sol.X.num_elements());
      std::copy(FnX.data(), FnX.data()+lr_sol.X.num_elements(), Xn.data()+2*lr_sol.X.num_elements());
    } else {
      #ifdef __CUDA__
      cudaMemcpy(Xn.data()+lr_sol.X.num_elements(), lr_sol.X.data(), sizeof(double)*lr_sol.X.num_elements(),cudaMemcpyDeviceToDevice);
      cudaMemcpy(Xn.data()+2*lr_sol.X.num_elements(), FnX.data(), sizeof(double)*lr_sol.X.num_elements(),cudaMemcpyDeviceToDevice);
      #endif
    }
    gs(Xn, Sbar, gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2]);

    // For V
    blas.matmul_transb(Vn,Sn,VnT); 
    L_step_rk4.rk4_L_rhs(VnT,v,D1,D2,FnV);
    FnV *= tau;

    Vn.update_shape({gi.dvv_mult,3*gi.r});
    if(Vn.sl == stloc::host){
      std::copy(lr_sol.V.data(), lr_sol.V.data()+lr_sol.V.num_elements(), Vn.data()+lr_sol.V.num_elements());
      std::copy(FnV.data(), FnV.data()+lr_sol.V.num_elements(), Vn.data()+2*lr_sol.V.num_elements());
    } else {
      #ifdef __CUDA__
      cudaMemcpy(Vn.data()+lr_sol.V.num_elements(), lr_sol.V.data(), sizeof(double)*lr_sol.V.num_elements(),cudaMemcpyDeviceToDevice);
      cudaMemcpy(Vn.data()+2*lr_sol.V.num_elements(), FnV.data(), sizeof(double)*lr_sol.V.num_elements(),cudaMemcpyDeviceToDevice);
      #endif
    }
    gs(Vn, Sbar, gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2]);

    // --- S step ---
    blas.matmul_transa(Xn,lr_sol.X,Mbar);
    Mbar *= gi.h_xx[0]*gi.h_xx[1]*gi.h_xx[2];
    blas.matmul_transa(Vn,lr_sol.V,Nbar);
    Nbar *= gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2];

    Sn.resize({3*gi.r,gi.r});
    blas.matmul(Mbar,lr_sol.S,Sn);
    blas.matmul_transb(Sn,Nbar,Sbar);

    for(int ii = 0; ii < 3; ii++){
      C1[ii].resize({3*gi.r,3*gi.r});
      C2[ii].resize({3*gi.r,3*gi.r});
      D1[ii].resize({3*gi.r,3*gi.r});
      D2[ii].resize({3*gi.r,3*gi.r});
    }
    compute_C.update_info(3*gi.r);
    compute_D.update_info(3*gi.r);
    compute_C(Vn, C1, C2, blas);
    compute_D(Xn, E, D1, D2, blas);

    S_step_rk4(tau, Sbar, C1, C2, D1, D2, nsteps_int);

    // Determine new rank
    svd(Sbar, UUs, VVs, sigma, blas);

    if(Sbar.sl == stloc::host){
      for(Index idx = 0; idx < 3*gi.r; idx++){
        double psum = 0;
        for(Index ii = idx; ii < 3*gi.r; ii++){
          psum += sigma(ii)*sigma(ii);
        }
        if (sqrt(psum) <= tol){
          newr = idx+1;
          break;
        }
      }
    } else {
      #ifdef __CUDA__
      cudaMemcpy(sigma_h.data(), sigma.data(), sizeof(double)*(3*gi.r),cudaMemcpyDeviceToHost);

      for(Index idx = 0; idx < 3*gi.r; idx++){
        double psum = 0;
        for(Index ii = idx; ii < 3*gi.r; ii++){
          psum += sigma_h(ii)*sigma_h(ii);
        }
        if (sqrt(psum) <= tol){
          newr = idx+1;
          break;
        }
      }
      #endif
    }

    if (3*newr > gi.rmax){
      cout << "Rank can't exceed maxrank, setting rank=rmax/3" << endl;
      newr = gi.rmax/3;
    }
    
    // Now truncate and do all the updates
    gi.update_rank(newr);
    lr_sol.update_info(gi.r);
    if (lr_sol.S.sl == stloc::host){
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index i=0; i < lr_sol.S.num_elements(); i++){
        Index idx_r = i%gi.r;
        Index idx_c = i/gi.r;
        if(idx_r == idx_c){
          lr_sol.S(idx_r,idx_c) = sigma(idx_r);
        } else {
          lr_sol.S(idx_r,idx_c) = 0.0;
        }
      }
    } else {
      #ifdef __CUDA__
      diag_S<<<gi.r,gi.r>>>(gi.r,lr_sol.S.data(),sigma.data());
      cudaDeviceSynchronize();
      #endif
    }
    UUs.update_shape({3*gi.r,gi.r});
    VVs.update_shape({3*gi.r,gi.r});
    blas.matmul(Xn,UUs,lr_sol.X);
    blas.matmul(Vn,VVs,lr_sol.V);

    K_step_rk4.update_info(gi.r);
    S_step_rk4.update_info(gi.r);
    L_step_rk4.update_info(gi.r);
    
    S_step_rk4_BUG1.update_info(gi.r);
    
    Xn.update_shape({gi.dxx_mult,gi.r});
    XnT.update_shape({gi.dxx_mult,gi.r});
    FnX.update_shape({gi.dxx_mult,gi.r});
    Vn.update_shape({gi.dvv_mult,gi.r});
    VnT.update_shape({gi.dvv_mult,gi.r});
    FnV.update_shape({gi.dvv_mult,gi.r});

    Sn.resize({gi.r,gi.r});
    M.resize({gi.r,gi.r});
    N.resize({gi.r,gi.r});
    Stmp.resize({gi.r,gi.r});
    Mbar.resize({3*gi.r,gi.r});
    Nbar.resize({3*gi.r,gi.r});
    Sbar.resize({3*gi.r,3*gi.r});

    efield.update_info(gi.r);
    compute_C.update_info(gi.r);
    compute_D.update_info(gi.r);
    for(int ii = 0; ii < 3; ii++){
      C1[ii].resize({gi.r,gi.r});
      C2[ii].resize({gi.r,gi.r});
      D1[ii].resize({gi.r,gi.r});
      D2[ii].resize({gi.r,gi.r});
    }
    UUs.resize({3*gi.r,3*gi.r});
    VVs.resize({3*gi.r,3*gi.r});
    sigma.resize({3*gi.r});
    #ifdef __CUDA__
    sigma_h.resize({3*gi.r});
    #endif

    t += tau;

  }

  ofstream h_rank_f("h_rank_bug_so_"+gstr+".data");
  for(Index i = 0; i < (Index)h_rank.size(); i++){
    h_rank_f << h_rank[i] << endl;
  }
  
  if (snapshots == 1){
    //multi_array<double,2> K({gi.dxx_mult,gi.r},sl);
    //multi_array<double,2> SOL({gi.dxx_mult,gi.dvv_mult},sl);
    //blas.matmul(lr_sol.X, lr_sol.S, K);
    //blas.matmul_transb(K, lr_sol.V, SOL);
    //dump("solution_bug_so_"+gstr+".data",SOL);
    
    mat TMPX({gi.N_xx[0],gi.r}, sl);
    mat TMPV({gi.N_vv[0],gi.r}, sl);
    mat TMP1({gi.N_xx[0],gi.r}, sl);
    mat SOL({gi.N_xx[0],gi.N_vv[0]}, sl);

    extract_rows(lr_sol.X, 0, gi.N_xx[0]-1, TMPX);
    extract_rows(lr_sol.V, pow(gi.N_vv[0],3)/2+pow(gi.N_vv[0],2)/2, pow(gi.N_vv[0],3)/2+pow(gi.N_vv[0],2)/2+gi.N_vv[0]-1, TMPV);
    blas.matmul(TMPX, lr_sol.S, TMP1);
    blas.matmul_transb(TMP1, TMPV, SOL);
    dump("slice_bug_so"+gstr+".data",SOL);
  }
}


int main(int argc, char** argv){

  cxxopts::Options options("vlasov_poisson", "3+3 dimensional dynamical low-rank Vlasov--Poisson solver");
  options.add_options()
  ("device", "Device the simulation is run on (can be either cpu or gpu)", cxxopts::value<string>()->default_value("cpu"))
  ("problem", "Initial value that is used in the simulation", cxxopts::value<string>()->default_value("ts"))
  ("nx", "Number of grid points in space (as a whitespace separated list)", cxxopts::value<string>()->default_value("32 32 32"))
  ("nv", "Number of grid points in velocity (as a whitespace separated list)", cxxopts::value<string>()->default_value("32 32 32"))
  ("final_time", "Time to which the simulation is run", cxxopts::value<double>()->default_value("40"))
  ("deltat", "The time step used in the simulation (usually denoted by \\Delta t or tau)", cxxopts::value<double>()->default_value("0.04"))
  ("r_init", "Initial rank of the simulation", cxxopts::value<int>()->default_value("25"))
  ("r_min", "Minimum rank of the simulation (ignored by BUG)", cxxopts::value<int>()->default_value("10"))
  ("r_max", "Maximum rank of the simulation", cxxopts::value<int>()->default_value("60"))
  ("err", "Error control", cxxopts::value<string>()->default_value("ee"))
  ("order", "Order of integrator", cxxopts::value<int>()->default_value("2"))
  ("tol_inc", "Absolute tolerance (increase) for error control/BUG tolerance", cxxopts::value<double>()->default_value("0.0001"))
  ("tol_inc_rel", "Relative tolerance (increase) for error control", cxxopts::value<double>()->default_value("-1"))
  ("tol_dec", "Tolerance (decrease) for error control (ignored by BUG)", cxxopts::value<double>()->default_value("0.0000000001"))
  ("tol_dec_rel", "Relative tolerance (decrease) for error control", cxxopts::value<double>()->default_value("-1"))
  ("omp_threads", "Number of OpenMP threads used in CPU parallelization (by default half the number of processes reported by the operating system are used)", cxxopts::value<int>()->default_value("-1"))
  ("snapshots", "Write final distribution function to disk", cxxopts::value<int>()->default_value("0"))
  ("h,help", "Help message")
  ;
  auto result = options.parse(argc, argv);

  if(result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  #ifndef __CUDA__
  CPU = true;
  #else
  string dev = result["device"].as<string>();
  if(dev == "cpu"){
    CPU = true;
    cout << "CPU SIMULATION" << endl;
  }else if(dev == "gpu"){
    CPU = false;
    cout << "GPU SIMULATION" << endl;
  }else {
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

  Index   r = result["r_init"].as<int>();
  Index   min_r = result["r_min"].as<int>();
  Index   max_r = result["r_max"].as<int>();
  double  tol_inc = result["tol_inc"].as<double>(); // absolute tolerance to increase rank or BUG tolerance
  double  tol_inc_rel = result["tol_inc"].as<double>(); // relative tolerance to increase rank
  if (tol_inc_rel == -1){
    tol_inc_rel = tol_inc/10.0;
  }
  double  tol_dec = result["tol_dec"].as<double>(); // absolute tolerance to decrease rank
  double  tol_dec_rel = result["tol_dec"].as<double>(); // relative tolerance to decrease rank
  if (tol_dec_rel == -1){
    tol_dec_rel = tol_dec/10.0;
  }
  double  final_time = result["final_time"].as<double>();
  double  tau = result["deltat"].as<double>();
  int order = result["order"].as<int>();
  Index snapshots = result["snapshots"].as<int>();
  string ec = result["err"].as<string>();


  int nsteps_int = 1;

  blas_ops blas(!CPU);

  // Setup the initial value
  string problem = result["problem"].as<string>();

  mfp<6> lim_xx;
  mfp<6> lim_vv;
  grid_info_reserve<3> gi(max_r,r, N_xx, N_vv);
  vec xx({gi.dxx_mult});
  vec vv({gi.dvv_mult});
  vector<const double*> X, V;
  
  if (problem == "ll"){
    //
    // Linear Landau damping
    //
    lim_xx = {0.0,4.0*M_PI,0.0,4.0*M_PI,0.0,4.0*M_PI};
    lim_vv = {-6.0,6.0,-6.0,6.0,-6.0,6.0};
    gi(lim_xx, lim_vv); 

    componentwise_vec_omp(gi.N_xx, [&xx, &gi](Index idx, array<Index,3> i) {
      double alpha1 = 0.01, alpha2 = 0.01, alpha3 = 0.01;
      double kappa1 = 0.5, kappa2 = 0.5, kappa3 = 0.5;
      mfp<3> x  = gi.x(i);
      xx(idx) = 1.0 + alpha1*cos(kappa1*x[0]) + alpha2*cos(kappa2*x[1]) + alpha3*cos(kappa3*x[2]);
    });

    componentwise_vec_omp(gi.N_vv, [&vv, &gi](Index idx, array<Index,3> i) {
      mfp<3> v  = gi.v(i);
      vv(idx) = (1.0/(sqrt(pow(2*M_PI,3)))) * exp(-(pow(v[0],2)+pow(v[1],2)+pow(v[2],2))/2.0);
    });

  } else if (problem == "nll"){
    //
    // Nonlinear Landau
    //
    lim_xx = {0.0,4.0*M_PI,0.0,4.0*M_PI,0.0,4.0*M_PI};
    lim_vv = {-6.0,6.0,-6.0,6.0,-6.0,6.0};
    gi(lim_xx, lim_vv); 

    componentwise_vec_omp(gi.N_xx, [&xx, &gi](Index idx, array<Index,3> i) {
      double alpha1 = 0.3, alpha2 = 0.3, alpha3 = 0.3;
      double kappa1 = 0.5, kappa2 = 0.5, kappa3 = 0.5;
      mfp<3> x  = gi.x(i);
      xx(idx) = 1.0 + alpha1*cos(kappa1*x[0]) + alpha2*cos(kappa2*x[1]) + alpha3*cos(kappa3*x[2]);
    });

    componentwise_vec_omp(gi.N_vv, [&vv, &gi](Index idx, array<Index,3> i) {
      mfp<3> v  = gi.v(i);
      vv(idx) = (1.0/(sqrt(pow(2*M_PI,3)))) * exp(-(pow(v[0],2)+pow(v[1],2)+pow(v[2],2))/2.0);
    });

  } else if (problem == "ts"){
    //
    // Two-stream instability
    //
    lim_xx = {0.0,10.0*M_PI,0.0,10.0*M_PI,0.0,10.0*M_PI};
    lim_vv = {-9.0,9.0,-9.0,9.0,-9.0,9.0};
    gi(lim_xx, lim_vv); 

    componentwise_vec_omp(gi.N_xx, [&xx, &gi](Index idx, array<Index,3> i) {
      double alpha1 = 0.001, alpha2 = 0.001, alpha3 = 0.001;
      double kappa1 = 1.0/5.0, kappa2 = 1.0/5.0, kappa3=1.0/5.0;
      mfp<3> x  = gi.x(i);
      xx(idx) = 1.0 + alpha1*cos(kappa1*x[0]) + alpha2*cos(kappa2*x[1]) + alpha3*cos(kappa3*x[2]);
    });

    componentwise_vec_omp(gi.N_vv, [&vv, &gi](Index idx, array<Index,3> i) {
      double v0 = 2.5, w0 = 0.0, u0=0.0;
      double v0b = -2.5, w0b = -2.25, u0b = -2.0;
      mfp<3> v  = gi.v(i);
      vv(idx) = (1.0/(sqrt(pow(8*M_PI,3)))) * (exp(-(pow(v[0]-v0,2))/2.0)+exp(-(pow(v[0]-v0b,2))/2.0))*(exp(-(pow(v[1]-w0,2))/2.0)+exp(-(pow(v[1]-w0b,2))/2.0))*(exp(-(pow(v[2]-u0,2))/2.0)+exp(-(pow(v[2]-u0b,2))/2.0));
    });

  } else if (problem == "bot"){
    //
    // Bump-on-tail
    //
    lim_xx = {0.0,20.0*M_PI/3.0,0.0,20.0*M_PI/3,0.0,20.0*M_PI/3};
    lim_vv = {-9.0,9.0,-9.0,9.0,-9.0,9.0};
    gi(lim_xx, lim_vv); 

    componentwise_vec_omp(gi.N_xx, [&xx, &gi](Index idx, array<Index,3> i) {
      double alpha = 0.03;
      double kappa = 0.3;
      mfp<3> x  = gi.x(i);
      xx(idx) = 1.0 + alpha*(cos(kappa*x[0]) + cos(kappa*x[1]) + cos(kappa*x[2]));
    });

    componentwise_vec_omp(gi.N_vv, [&vv, &gi](Index idx, array<Index,3> i) {
      mfp<3> v  = gi.v(i);
      vv(idx) = (1.0/(sqrt(pow(2*M_PI,3)))) * 
                  exp(-(pow(v[1],2)+pow(v[2],2))/2.0) *
                  (0.9*exp(-pow(v[0],2)/2)+0.2*exp(-2*pow(v[0]-4.5,2)));
    });

  } else {
    cout << "ERROR: problem with name " << problem << " is not supported" << endl;
    exit(1);
  }

  X.push_back(xx.begin());
  V.push_back(vv.begin());

  #ifdef __MKL__
    cout << "MKL ENABLED" << endl;
  #endif
  cout << "Simulation: " << problem << endl;
  cout << "Dof in space: " << gi.N_xx[0] << " " << gi.N_xx[1] << " " << gi.N_xx[2] << " " << endl;
  cout << "Dof in velocity: " << gi.N_vv[0] << " " << gi.N_vv[1] << " " << gi.N_vv[2] << " " << endl;
  cout << "Final time: " << final_time << endl;
  cout << "Deltat: " << tau << endl;
  cout << "Error control: " << ec << endl;
  cout << "Order of integrator: " << order << endl;
  cout << "Tolerance incr (abs): " << tol_inc << endl;
  cout << "Tolerance incr (rel): " << tol_inc_rel << endl;
  cout << "Tolerance decr (abs): " << tol_dec << endl;
  cout << "Tolerance decr (rel): " << tol_dec_rel << endl;
  cout << "Initial rank: " << gi.r << endl;
  cout << "Min rank: " << min_r << endl;
  cout << "Max rank: " << max_r << endl;


  gt::start("Total WC time");
  if (ec == "bug"){
    // Remark that gi.rmax is the maximum number of columns allowed. Therefore, since
    // in BUG we look for the rank in a space of size 2*r or 3*r, the maximum rank allowed
    // is gi.rmax/2 or gi.rmax/3
    if (order == 1){
      BUG_adapt(final_time, tau, nsteps_int, gi, X, V, tol_inc, max_r, snapshots, blas);
    } else if (order == 2){
      BUG_so_adapt(final_time, tau, nsteps_int, gi, X, V, tol_inc, max_r, snapshots, blas);
    } else {
      cout << "ERROR: order " << order << " is not supported." << endl;
    }
  } else if (ec == "ee" || ec == "ef" || ec == "f"){
    if (order == 1){
      PS_adapt(final_time, tau, nsteps_int, gi, X, V, tol_inc, tol_inc_rel, tol_dec, tol_dec_rel, min_r, max_r, ec, snapshots, blas);
    } else if (order == 2){
      PS_so_adapt(final_time, tau, nsteps_int, gi, X, V, tol_inc, tol_inc_rel, tol_dec, tol_dec_rel, min_r, max_r, ec, snapshots, blas);
    } else {
      cout << "ERROR: order " << order << " is not supported." << endl;
    }
  } else {
    cout << "ERROR: error control with name " << ec << " is not supported." << endl;
    exit(1);
  }
  gt::stop("Total WC time");

  cout << gt::sorted_output() << endl;


/*
  // For error check
  snapshots = 1;
  //tau = 0.000033333333333333333; // reference

  tau = 0.01;
  //tau = 0.005;
  //tau = 0.0033333333333333333333;
  //tau = 0.0025;
  //tau = 0.002;
  //tau = 0.0016666666666666666666;
  //tau = 0.0014285714285714285714;
  //tau = 0.00125;
  //tau = 0.0011111111111111111111;
  //tau = 0.001;
  //BUG_adapt(final_time, tau, nsteps_int, gi, X, V, tol_inc, max_r, snapshots, blas);
  //BUG_so_adapt(final_time, tau, nsteps_int, gi, X, V, tol_inc, max_r, snapshots, blas);
  //PS_adapt(final_time, tau, nsteps_int, gi, X, V, tol_inc, tol_inc_rel, tol_dec, tol_dec_rel, min_r, max_r, ec, snapshots, blas);
  PS_so_adapt(final_time, tau, nsteps_int, gi, X, V, tol_inc, tol_inc_rel, tol_dec, tol_dec_rel, min_r, max_r, ec, snapshots, blas);
*/

  return 0;
}
