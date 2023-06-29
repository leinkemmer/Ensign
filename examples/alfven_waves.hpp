#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/timer.hpp>
#include <generic/fft.hpp>
#include <iomanip>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

template<size_t d> using mind = array<Index,d>;
template<size_t d> using mfp  = array<double,d>;
using mat  = multi_array<double,2>;
using cmat  = multi_array<complex<double>,2>;
using ten3  = multi_array<double,3>;
using vec  = multi_array<double,1>;
using cvec  = multi_array<complex<double>,1>;

Index freq(Index k, Index n) {
  if(k < n/2)
    return k;
  else if(k == n/2)
    return 0;
  else
    return k-n;
}

enum class discretization { fft, lw };

template<size_t d>
struct grid_info {
  Index r;
  mind<d>  N_xx, N_zv;
  mfp<2*d> lim_xx, lim_zv;
  mfp<d>   h_xx, h_zv;
  Index dxx_mult, dzv_mult, dxxh_mult, dzvh_mult;
  double M_e, C_P, C_A;
  discretization discr;

  bool debug_adv_z, debug_adv_v, debug_adv_vA;

  grid_info(Index _r, mind<d> _N_xx, mind<d> _N_zv, mfp<2*d> _lim_xx, mfp<2*d> _lim_zv, double _M_e, double _C_P, double _C_A, discretization _discr)
    : r(_r), N_xx(_N_xx), N_zv(_N_zv), lim_xx(_lim_xx), lim_zv(_lim_zv), M_e(_M_e), C_P(_C_P), C_A(_C_A), discr(_discr), debug_adv_z(true), debug_adv_v(true), debug_adv_vA(true) {

    // compute h_xx and h_zv
    for(int ii = 0; ii < 2; ii++){
      Index jj = 2*ii;
      h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
      h_zv[ii] = (lim_zv[jj+1]-lim_zv[jj])/ N_zv[ii];
    }
    
    dxx_mult  = N_xx[0]*N_xx[1];
    dzv_mult  = N_zv[0]*N_zv[1];
    dxxh_mult = N_xx[1]*(N_xx[0]/2 + 1);
    dzvh_mult = N_zv[1]*(N_zv[0]/2 + 1);
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
      stride *= N_zv[k];
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
    return lim_zv[2*k] + i*h_zv[k];
  }

  mfp<d> v(mind<d> i) const {
    mfp<d> out;
    for(size_t k=0;k<d;k++)
      out[k] = v(k, i[k]);
    return out;
  }
};

template<size_t d>
grid_info<d> modify_r(grid_info<d> gi, Index r) {
  gi.r = r;
  return gi;
}


// Note that using a std::function object has a non-negligible performance overhead
template<class func>
void componentwise_vec_omp(const mind<2>& N, func F) {
  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < N[1]; j++){
    for(Index i = 0; i < N[0]; i++){
      Index idx = i+j*N[0];
      F(idx, {i,j});
    }
  }
}

template<class func>
void componentwise_mat_omp(Index r, const mind<2>& N,  func F) {
  #ifdef __OPENMP__
  #pragma omp parallel for collapse(2)
  #endif
  for(int rr = 0; rr < r; rr++){
    for(Index j = 0; j < N[1]; j++){
      for(Index i = 0; i < N[0]; i++){
        Index idx = i+j*N[0];
        F(idx, {i,j}, rr);
      }
    }
  }
}


template<class func>
void componentwise_mat_fourier_omp(Index r, const mind<2>& N,  func F) {
  #ifdef __OPENMP__
  #pragma omp parallel for collapse(2)
  #endif
  for(int rr = 0; rr < r; rr++){
    for(Index j = 0; j < N[1]; j++){
      for(Index i = 0; i < (N[0]/2 + 1); i++){
        Index idx = i+j*(N[0]/2+1);
        F(idx, {i,j}, rr);
      }
    }
  }
}


void deriv_z(const mat& in, mat& out, const grid_info<2>& gi) {
  Index r = in.shape()[1];
  if(gi.discr == discretization::lw) {
    mat tmp(out); // in order to allow for in and out to be the same
    
    // finite differences
    if(in.shape()[0] == gi.N_zv[0] && out.shape()[0] == gi.N_zv[0]) {
      // only depends on z
      for(Index ir=0;ir<r;ir++) {
        for(Index iz=0;iz<gi.N_zv[0];iz++)
          tmp(iz, ir) = (in((iz+1)%gi.N_zv[0], ir) - in((iz-1+gi.N_zv[0])%gi.N_zv[0], ir))/(2.0*gi.h_zv[0]);
      }
    } else if(in.shape()[0] == gi.N_zv[0] && out.shape()[0] != gi.N_zv[0]) {
      // out depends on both z and v, but in does only depend on z
      for(Index ir=0;ir<r;ir++) {
        for(Index iv=0;iv<gi.N_zv[1];iv++) {
          for(Index iz=0;iz<gi.N_zv[0];iz++)
            tmp(gi.lin_idx_v({iz,iv}), ir) = (in((iz+1)%gi.N_zv[0], ir) - in((iz-1+gi.N_zv[0])%gi.N_zv[0], ir))/(2.0*gi.h_zv[0]);
        }
      }
    } else {
      // both input and output depends on both z and v
      for(Index ir=0;ir<r;ir++) {
        for(Index iv=0;iv<gi.N_zv[1];iv++) {
          for(Index iz=0;iz<gi.N_zv[0];iz++)
            tmp(gi.lin_idx_v({iz,iv}), ir) = (in(gi.lin_idx_v({(iz+1)%gi.N_zv[0],iv}), ir) - in(gi.lin_idx_v({(iz-1+gi.N_zv[0])%gi.N_zv[0],iv}), ir))/(2.0*gi.h_zv[0]);
        }
      }
    }

    out = tmp;
  } else {

    // FFT
    if(in.shape()[0] == gi.N_zv[0]) {
      cmat hat({gi.N_zv[0]/2+1, r});
      fft1d<2> fft({gi.N_zv[0]}, (mat&)in, hat);

      fft.forward((mat&)in, hat);

      double ncz = 1.0/double(gi.N_zv[0]);
      for(Index ir=0;ir<r;ir++) {
        for(Index iz=0;iz<gi.N_zv[0]/2+1;iz++) {
          complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(gi.lim_zv[1]-gi.lim_zv[0])*iz);
          hat(iz, ir) *= lambdaz*ncz;
        }
      }

      if(out.shape()[0] == gi.N_zv[0]) {
        // both input and output only depend on z
        fft.backward(hat, out);
      } else {
        // output depends on z and v, but input only depends on z
        mat tmp({gi.N_zv[0], r});
        fft.backward(hat, tmp);
        for(Index ir=0;ir<r;ir++) {
          for(Index iv=0;iv<gi.N_zv[1];iv++) {
            for(Index iz=0;iz<gi.N_zv[0];iz++)
              out(gi.lin_idx_v({iz,iv}), ir) = tmp(iz, ir);
          }
        }
      }
    } else {
      // both input and output depend on z and v
      cmat hat({gi.dzvh_mult, r});
      fft2d<2> fft(gi.N_zv, (mat&)in, hat);

      fft.forward((mat&)in, hat);

      double nczv = 1.0/double(gi.N_zv[0]*gi.N_zv[1]);
      componentwise_mat_fourier_omp(r, gi.N_zv, [nczv,&gi,&hat](Index idx, mind<2> i, Index r) {
        complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(gi.lim_zv[1]-gi.lim_zv[0])*i[0]);
            
        hat(idx, r) *= lambdaz*nczv;
      });

      fft.backward(hat, out);
    }
  }
}

void deriv_v(const mat& in, mat& out, const grid_info<2>& gi) {
  if(gi.discr == discretization::lw) {

    // finite differences
    // depends on both z and v
    for(Index ir=0;ir<gi.r;ir++) {
      for(Index iv=0;iv<gi.N_zv[1];iv++) {
        for(Index iz=0;iz<gi.N_zv[0];iz++)
          out(gi.lin_idx_v({iz,iv}), ir) = (in(gi.lin_idx_v({iz,(iv+1)%gi.N_zv[1]}), ir) - in(gi.lin_idx_v({iz, (iv-1+gi.N_zv[1])%gi.N_zv[1]}), ir))/(2.0*gi.h_zv[1]);
      }
    }

  } else {

    // FFT
    cmat hat({gi.dzvh_mult, gi.r});
    fft2d<2> fft(gi.N_zv, (mat&)in, hat);

    fft.forward((mat&)in, hat);

    double nczv = 1.0/double(gi.N_zv[0]*gi.N_zv[1]);
    componentwise_mat_fourier_omp(gi.r, gi.N_zv, [nczv,&gi,&hat](Index idx, mind<2> i, Index r) {
      Index mult_j = freq(i[1], gi.N_zv[1]);
      complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(gi.lim_zv[3]-gi.lim_zv[2])*mult_j);
          
      hat(idx, r) *= lambdav*nczv;
    });

    fft.backward(hat, out);
  }

}



void deriv_x(const mat& in, mat& out, const grid_info<2>& gi) {
  if(gi.discr == discretization::lw) {
  
    // centered differences
    for(Index ir=0;ir<gi.r;ir++) {
      for(Index ix=0;ix<gi.N_xx[0];ix++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++)
          out(gi.lin_idx_x({ix,iy}), ir) = (in(gi.lin_idx_x({(ix+1)%gi.N_xx[0],iy}), ir) - in(gi.lin_idx_x({(ix-1+gi.N_xx[0])%gi.N_xx[0], iy}), ir))/(2.0*gi.h_xx[0]);
      }
    }

  } else {

    // FFT
    cmat hat({gi.dxxh_mult, gi.r});
    fft2d<2> fft(gi.N_xx, (mat&)in, hat);

    fft.forward((mat&)in, hat);

    double ncxx = 1.0/double(gi.N_xx[0]*gi.N_xx[1]);
    componentwise_mat_fourier_omp(gi.r, gi.N_xx, [ncxx,&gi,&hat](Index idx, mind<2> i, Index r) {
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[1]-gi.lim_xx[0])*i[0]);
          
      hat(idx, r) *= lambdax*ncxx;
    });

    fft.backward(hat, out);
  }
}

void deriv_y(const mat& in, mat& out, const grid_info<2>& gi) {
  if(gi.discr == discretization::lw) {

    // centered differences
    for(Index ir=0;ir<gi.r;ir++) {
      for(Index ix=0;ix<gi.N_xx[0];ix++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++)
          out(gi.lin_idx_x({ix,iy}), ir) = (in(gi.lin_idx_x({ix,(iy+1)%gi.N_xx[1]}), ir) - in(gi.lin_idx_x({ix, (iy-1+gi.N_xx[1])%gi.N_xx[1]}), ir))/(2.0*gi.h_xx[1]);
      }
    }

  } else {

    // FFT
    cmat hat({gi.dxxh_mult, gi.r});
    fft2d<2> fft(gi.N_xx, (mat&)in, hat);

    fft.forward((mat&)in, hat);

    double ncxx = 1.0/double(gi.N_xx[0]*gi.N_xx[1]);
    componentwise_mat_fourier_omp(gi.r, gi.N_xx, [ncxx,&gi,&hat](Index idx, mind<2> i, Index r) {
      Index mult_j = freq(i[1], gi.N_xx[1]);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[3]-gi.lim_xx[2])*mult_j);
          
      hat(idx, r) *= lambday*ncxx;
    });

    fft.backward(hat, out);
  }
}



struct compute_coeff {
  mat compute_C1(const mat& V, const blas_ops& blas) {
    gt::start("C1");

    ptw_mult_row(V,v,Vtmp1); // multiply by V
    deriv_z(V, Vtmp2, gi);

    mat C1({gi.r, gi.r});
    coeff(Vtmp1, Vtmp2, gi.h_zv[0]*gi.h_zv[1], C1, blas);

    gt::stop("C1");
    return C1;
  }

  ten3 compute_C2(const mat& Vf, const mat& Vphi, const blas_ops& blas) {
    gt::start("C2");

    deriv_v(Vf, Vtmp1, gi);
    deriv_z(Vphi, Vtmp_fields, gi);

    ten3 C2({gi.r,gi.r,r_fields});
    coeff(Vf, Vtmp1, Vtmp_fields, gi.h_zv[0]*gi.h_zv[1], C2, blas);
    
    gt::stop("C2");
    return C2;
  }
  
  ten3 compute_C3(const mat& Vf, const mat& VdtA, const blas_ops& blas) {
    gt::start("C3");

    deriv_v(Vf, Vtmp1, gi);

    // Vtmp2 has to be enlarged to the full zv space such that the coeff function
    // can be used.
    componentwise_mat_omp(r_fields, gi.N_zv, [this, &VdtA](Index idx, mind<2> i, Index r) {
      Vtmp_fields(idx, r) = VdtA(i[0], r);
    });

    ten3 C3({gi.r,gi.r,r_fields});
    coeff(Vf, Vtmp1, Vtmp_fields, gi.h_zv[0]*gi.h_zv[1], C3, blas);

    gt::stop("C3");
    return C3;
  }
  
  ten3 compute_D2(const mat& Xf, const mat& Xphi, const blas_ops& blas) {
    gt::start("D2");

    ten3 D2({gi.r,gi.r,r_fields});
    coeff(Xf, Xf, Xphi, gi.h_xx[0]*gi.h_xx[1], D2, blas);

    gt::stop("D2");
    return D2;
  }
  
  ten3 compute_D3(const mat& Xf, const mat& XdtA, const blas_ops& blas) {
    gt::start("D3");

    ten3 D3({gi.r,gi.r,r_fields});
    coeff(Xf, Xf, XdtA, gi.h_xx[0]*gi.h_xx[1], D3, blas);

    gt::stop("D3");
    return D3;
  }

  ten3 compute_e(const ten3& D2, const mat& Lphi) {
    gt::start("e");

    deriv_z(Lphi, dzLphi, gi);

    ten3 e({gi.N_zv[0], gi.r, gi.r});
    for(Index i=0;i<gi.r;i++) {
      for(Index k=0;k<gi.r;k++) {
        for(Index iz=0;iz<gi.N_zv[0];iz++) {
          double val = 0.0;
          for(Index m=0;m<r_fields;m++)
            val += D2(i, k, m)*dzLphi(iz, m);
          e(iz, i, k) = val;
        }
      }
    }

    gt::stop("e");
    return e;
  }
  
  ten3 compute_eA(const ten3& D3, const mat& LdtA) {
    ten3 eA({gi.N_zv[0], gi.r, gi.r});
    for(Index i=0;i<gi.r;i++) {
      for(Index k=0;k<gi.r;k++) {
        for(Index iz=0;iz<gi.N_zv[0];iz++) {
          double val = 0.0;
          for(Index m=0;m<r_fields;m++)
            val += D3(i, k, m)*LdtA(iz, m);
          eA(iz, i, k) = val;
        }
      }
    }
    return eA;
  }

  compute_coeff(grid_info<2> _gi, Index _r_fields=-1) : gi(_gi) {
    if(_r_fields == -1)
      r_fields = gi.r;
    else
      r_fields = _r_fields;

    Vtmp1.resize({gi.dzv_mult,gi.r});
    Vtmp2.resize({gi.dzv_mult,gi.r});
    Vtmp_fields.resize({gi.dzv_mult,r_fields});
    dzLphi.resize({gi.N_zv[0], r_fields});

    v.resize({gi.dzv_mult});
    componentwise_vec_omp(gi.N_zv, [this](Index idx, mind<2> i) {
      v(idx) = gi.v(1, i[1]);
    });
  }

private:
  grid_info<2> gi;
  Index r_fields;
  mat Vtmp1, Vtmp2, Vtmp_fields;
  mat dzLphi;
  vec v;
};



void rk4(double tau, mat& U, std::function<void(const mat&, mat&)> rhs) {
  gt::start("rk4");
  mat k1(U);
  mat k2(U);
  mat k3(U);
  mat k4(U);
  mat tmp(U);
  mat in = U;

  // k1
  gt::start("rk4_rhs");
  rhs(in, k1);
  gt::stop("rk4_rhs");

  // k2
  tmp = k1;
  tmp *= 0.5*tau;
  tmp += in;
  gt::start("rk4_rhs");
  rhs(tmp, k2);
  gt::stop("rk4_rhs");

  // k3
  tmp = k2;
  tmp *= 0.5*tau;
  tmp += in;
  gt::start("rk4_rhs");
  rhs(tmp, k3);
  gt::stop("rk4_rhs");

  // k4
  tmp = k3;
  tmp *= tau;
  tmp += in;
  gt::start("rk4_rhs");
  rhs(tmp, k4);
  gt::stop("rk4_rhs");
  
  k1 *= 1.0/6.0*tau;
  U += k1;
  k2 *= 1.0/6.0*tau*2.0; 
  U += k2;
  k3 *= 1.0/6.0*tau*2.0;
  U += k3;
  k4 *= 1.0/6.0*tau;
  U += k4;
  gt::stop("rk4");
}




struct PS_K_step {

  void operator()(double tau, mat& K, const mat& Kphi, const mat& KdtA, const mat& C1, const ten3& C2, const ten3& C3, const blas_ops& blas) {

    rk4(tau, K, [this, &C1, &C2, &C3, &Kphi, &KdtA, &blas](const mat& K, mat& Kout) {

      blas.matmul_transb(K, C1, Kout);
      Kout *= -double(gi.debug_adv_z);

      gt::start("K2");
      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(Index j=0;j<gi.r;j++) {
        for(Index i=0;i<gi.dxx_mult;i++) {
          double s = Kout(i,j);
          for(Index l=0;l<gi.r;l++) {
            for(Index n=0;n<gi.r;n++) {
              s -=   double(gi.debug_adv_v)/gi.M_e*C2(j,l,n)*Kphi(i, n)*K(i, l)
                   + double(gi.debug_adv_vA)/gi.M_e*C3(j,l,n)*KdtA(i, n)*K(i, l);
            }
          }
          Kout(i, j) = s;
        }
      }
      gt::stop("K2");
    });
  }

  PS_K_step(const grid_info<2>& _gi, const blas_ops& _blas) : gi(_gi), blas(_blas) {}

private:
  grid_info<2> gi;
  const blas_ops& blas;
};

struct PS_S_step {

  void operator()(double tau, mat& S, const mat& Sphi, const mat& SdtA, const mat& C1, const ten3& C2, const ten3& C3, const ten3& D2, const ten3& D3) {
    rk4(tau, S, [this, &C1, &C2, &C3, &D2, &D3, &Sphi, &SdtA](const mat& S, mat& Sout) {
      blas.matmul_transb(S, C1, Sout);
      Sout *= double(gi.debug_adv_z);
      
      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(Index i=0;i<gi.r;i++) {
        for(Index j=0;j<gi.r;j++) {
          double s = Sout(i, j);
          for(Index l=0;l<gi.r;l++) {
            for(Index n=0;n<SdtA.shape()[1];n++) {
              for(Index m=0;m<SdtA.shape()[0];m++) {
                for(Index k=0;k<gi.r;k++) {
                  s += double(gi.debug_adv_v)/gi.M_e*D2(i,k,m)*Sphi(m,n)*S(k,l)*C2(j,l,n)
                               +double(gi.debug_adv_vA)/gi.M_e*D3(i,k,m)*SdtA(m,n)*S(k,l)*C3(j,l,n);
                }
              }
            }
          }
          Sout(i, j) = s;
        }
      }
      
    });
  }

  PS_S_step(const grid_info<2>& _gi, const blas_ops& _blas) : gi(_gi), blas(_blas) {}

private:
  grid_info<2> gi;
  const blas_ops& blas;
};

struct PS_L_step {

  void operator()(double tau, mat& L, const ten3& e, const ten3& eA) {
    // Here use use a splitting scheme between the advection in z and the advection in v

    vec tmp({gi.N_zv[0]});
    cvec Lhat({gi.N_zv[0]/2+1});
    fft1d<1> fft({gi.N_zv[0]}, tmp, Lhat);

    mat M({gi.N_zv[1], gi.r});
    mat Mtmp({gi.N_zv[1], gi.r});
    cmat Mhat({gi.N_zv[1]/2+1, gi.r});
    fft1d<2> fft2({gi.N_zv[1]}, M, Mhat);

    if(gi.debug_adv_z) {

      if(gi.discr == discretization::lw) {
        // the term -v \partial_z L using a Lax-Wendroff scheme
        componentwise_mat_omp(gi.r, gi.N_zv, [this, tau, &L](Index idx, mind<2> i, Index r) {
          Index idx_p1 = gi.lin_idx_v({(i[0]+1)%gi.N_zv[0],i[1]});
          Index idx_m1 = gi.lin_idx_v({(i[0]-1+gi.N_zv[0])%gi.N_zv[0],i[1]});
          double v = gi.v(1, i[1]);
          Ltmp(idx, r) = L(idx,r) - 0.5*tau/gi.h_zv[0]*v*(L(idx_p1, r) - L(idx_m1, r))
                          +0.5*pow(tau,2)/pow(gi.h_zv[0],2)*pow(v,2)*(L(idx_p1,r)-2.0*L(idx,r)+L(idx_m1,r));
        });

      } else { // fft

        // the term -v \partial_z L using FFT
        double ncz = 1.0/double(gi.N_zv[0]);
        for(Index ir=0;ir<gi.r;ir++) {
          for(Index iv=0;iv<gi.N_zv[1];iv++) {

            for(Index iz=0;iz<gi.N_zv[0];iz++) {
              tmp(iz) = L(gi.lin_idx_v({iz,iv}), ir);
            }

            fft.forward((vec&)tmp, Lhat);

            double v = gi.v(1, iv);
            for(Index iz=0;iz<gi.N_zv[0]/2+1;iz++) {
              complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(gi.lim_zv[1]-gi.lim_zv[0])*iz);
              Lhat(iz) *= exp(-tau*lambdaz*v)*ncz;
            }

            fft.backward(Lhat, tmp);

            for(Index iz=0;iz<gi.N_zv[0];iz++)
              Ltmp(gi.lin_idx_v({iz,iv}), ir) = tmp(iz);
          }
        }
      }

    } else {
      Ltmp = L;
    }

    if(gi.debug_adv_v || gi.debug_adv_vA) {
      mat ez({gi.r,gi.r}), T({gi.r, gi.r});
      vec lambda({gi.r});
      mat Mout({gi.N_zv[1], gi.r});

      for(Index iz=0;iz<gi.N_zv[0];iz++) {

        for(Index k2=0;k2<gi.r;k2++) {
          for(Index k1=0;k1<gi.r;k1++) {
            ez(k1, k2) = double(gi.debug_adv_v)*e(iz, k1, k2) + double(gi.debug_adv_vA)*eA(iz, k1, k2);
          }
        }

        schur(ez, T, lambda);

        // compute M from L
        for(Index ir=0;ir<gi.r;ir++) {
          for(Index iv=0;iv<gi.N_zv[1];iv++) {
            double val = 0.0;
            for(Index n=0;n<gi.r;n++)
              val += T(n, ir)*Ltmp(gi.lin_idx_v({iz,iv}), n);
            M(iv, ir) = val;
          }
        }


        // solve equation in diagonalized form
        if(gi.discr == discretization::lw) {

          // Lax-Wendroff scheme
          for(Index ir=0;ir<gi.r;ir++) {
            for(Index iv=0;iv<gi.N_zv[1];iv++) {
              Index iv_p1 = (iv+1)%gi.N_zv[1];
              Index iv_m1 = (iv-1+gi.N_zv[1])%gi.N_zv[1];
              Mout(iv, ir) = M(iv, ir) - 0.5*tau/gi.h_zv[1]*lambda(ir)/gi.M_e*(M(iv_p1, ir) - M(iv_m1, ir))
                                      + 0.5*pow(tau,2)/pow(gi.h_zv[1],2)*pow(lambda(ir)/gi.M_e,2)*(M(iv_p1,ir)-2.0*M(iv,ir)+M(iv_m1,ir));

            }
          }

        } else { // fft

          // FFT based scheme
          fft2.forward(M, Mhat);
          double ncv = 1.0/double(gi.N_zv[1]);
          for(Index ir=0;ir<gi.r;ir++) {
            for(Index iv=0;iv<gi.N_zv[1]/2+1;iv++) {
                complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(gi.lim_zv[3]-gi.lim_zv[2])*iv);
                Mhat(iv, ir) *= exp(-tau*lambdav*lambda(ir)/gi.M_e)*ncv;
              }
          }
          fft2.backward(Mhat, Mout);

        }

        // compute L from M
        for(Index ir=0;ir<gi.r;ir++) {
          for(Index iv=0;iv<gi.N_zv[1];iv++) {
            double val = 0.0;
            for(Index n=0;n<gi.r;n++)
              val += T(ir, n)*Mout(iv, n);
            L(gi.lin_idx_v({iz,iv}), ir) = val;
          }
        }

      }

    } else {
      L = Ltmp;
    }
  }


  PS_L_step(const grid_info<2>& _gi, const blas_ops& _blas) : gi(_gi), blas(_blas), schur(_gi.r) {
    Ltmp.resize({gi.dzv_mult, gi.r});
  }

private:
  grid_info<2> gi;
  const blas_ops& blas;
  diagonalization schur;
  mat Ltmp;
};

// integrate over v (but not z)
void integrate_v(const mat& V, mat& intV, const grid_info<2>& gi) {
  for(Index k=0;k<gi.r;k++)
    for(Index i=0;i<gi.N_zv[0];i++)
      intV(i, k) = 0.0;

  for(Index r=0;r<gi.r;r++) {
    for(Index iz=0;iz<gi.N_zv[0];iz++) {
      double s = 0.0;
      for(Index iv=0;iv<gi.N_zv[1];iv++)
        s += V(gi.lin_idx_v({iz,iv}), r)*gi.h_zv[1];
      intV(iz, r) = s;
    }
  }
}

void integrate_mulv_v(const mat& V, mat& intV, const grid_info<2>& gi) {
  for(Index k=0;k<gi.r;k++)
    for(Index i=0;i<gi.N_zv[0];i++)
      intV(i, k) = 0.0;


  for(Index r=0;r<gi.r;r++) {
    for(Index iz=0;iz<gi.N_zv[0];iz++) {
      for(Index iv=0;iv<gi.N_zv[1];iv++)
        intV(iz, r) += -gi.v(1, iv)*V(gi.lin_idx_v({iz,iv}), r)*gi.h_zv[1];

    }
  }
}

struct scalar_potential {

  void operator()(const mat& Kf, const mat& Vf, mat& Kphi, mat& Vphi, mat* Kmrho=nullptr) {
    // compute the basis of <V_j^f>_v
    integrate_v(Vf, intVf, gi);
    gs(intVf, intVf_R, ip_z);

    // expand 1 in th/dtA_at basis
    integrate(intVf, gi.h_zv[0], expansion_1, blas);

    // Construct the K for the rhs of the quasi-neutrality equation
    componentwise_mat_omp(gi.r, gi.N_xx, [this, &Kf, &Kmrho](Index idx, mind<2> i, Index j) {
      double val = 0.0;
      for(Index k=0;k<gi.r;k++)
        val += Kf(idx, k)*intVf_R(j, k);
      Krhs(idx, j) =  gi.C_P*(expansion_1(j) - val);
      if(Kmrho != nullptr)
        Kmrho->operator()(idx, j) = val;
    });

    // Solve the system
    Vphi = intVf;

    if(fft == nullptr)
      fft = make_unique_ptr<fft2d<2>>(gi.N_xx, Krhs, Kphihat);

    fft->forward(Krhs, Kphihat);

    double ncxx = 1.0/double(gi.N_xx[0]*gi.N_xx[1]);
    componentwise_mat_fourier_omp(gi.r, gi.N_xx, [this, ncxx](Index idx, mind<2> i, Index r) {
      Index mult_j = freq(i[1], gi.N_xx[1]);
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[1]-gi.lim_xx[0])*i[0]);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[3]-gi.lim_xx[2])*mult_j);
          
      Kphihat(idx, r) *= (i[0]==0 && mult_j==0) ? 0.0 : -1.0/(pow(lambdax,2) + pow(lambday,2))*ncxx;
    });

    fft->backward(Kphihat, Kphi);
  } 
  
  scalar_potential(const grid_info<2>& _gi, const blas_ops& _blas) : gi(_gi), blas(_blas), gs(&_blas) {
    intVf.resize({gi.N_zv[0], gi.r});
    intVf_R.resize({gi.r, gi.r});
    expansion_1.resize({gi.r});
    Krhs.resize({gi.dxx_mult, gi.r});
    Kphihat.resize({gi.dxxh_mult, gi.r});
    ip_z = inner_product_from_const_weight(gi.h_zv[0], gi.N_zv[0]);
  }

private:
  mat intVf, intVf_R, Krhs;
  cmat Kphihat;
  vec expansion_1;
  grid_info<2> gi;
  const blas_ops& blas;
  gram_schmidt gs;
  std::unique_ptr<fft2d<2>> fft;
  std::function<double(double*,double*)> ip_z;
};


double mass(const mat& K, const mat& V, const grid_info<2>& gi, const blas_ops& blas) {
  vec intK({gi.r});
  integrate(K, gi.h_xx[0]*gi.h_xx[1], intK, blas);

  vec intV({gi.r});
  integrate(V, gi.h_zv[0]*gi.h_zv[1], intV, blas);

  double val = 0.0;
  for(Index i=0;i<gi.r;i++)
    val += intK(i)*intV(i);
  return val;
}

double momentum(const mat& K, const mat& V, const grid_info<2>& gi, const blas_ops& blas) {
  vec intK({gi.r});
  integrate(K, gi.h_xx[0]*gi.h_xx[1], intK, blas);

  vec intvV({gi.r});
  mat vV({gi.dzv_mult, gi.r});
  componentwise_mat_omp(gi.r, gi.N_zv, [&vV, &V, &gi](Index idx, mind<2> i, Index r) {
    vV(idx, r) = gi.v(1,i[1])*V(idx, r);
  });
  integrate(vV, gi.h_zv[0]*gi.h_zv[1], intvV, blas);

  double val = 0.0;
  for(Index i=0;i<gi.r;i++)
    val += intK(i)*intvV(i);
  return val;
}

double kinetic_energy(const mat& K, const mat& V, const grid_info<2>& gi, const blas_ops&blas) {
  vec intK({gi.r});
  integrate(K, gi.h_xx[0]*gi.h_xx[1], intK, blas);

  vec intvsqV({gi.r});
  mat vsqV({gi.dzv_mult, gi.r});
  componentwise_mat_omp(gi.r, gi.N_zv, [&vsqV, &V, &gi](Index idx, mind<2> i, Index r) {
    vsqV(idx, r) = pow(gi.v(1,i[1]), 2)*V(idx, r);
  });
  integrate(vsqV, gi.h_zv[0]*gi.h_zv[1], intvsqV, blas);

  double val = 0.0;
  for(Index i=0;i<gi.r;i++)
    val += intK(i)*intvsqV(i);
  return 0.5*gi.M_e*val;
}

double electric_energy(const mat& Kphi, const grid_info<2>& gi, const blas_ops& blas) {
  mat dx({gi.r,gi.r}), dy({gi.r,gi.r});

  mat dxKphi({gi.dxx_mult, gi.r});
  deriv_x(Kphi, dxKphi, gi);
  coeff(dxKphi, dxKphi, gi.h_xx[0]*gi.h_xx[1], dx, blas);

  mat dyKphi({gi.dxx_mult, gi.r});
  deriv_y(Kphi, dyKphi, gi);
  coeff(dyKphi, dyKphi, gi.h_xx[0]*gi.h_xx[1], dy, blas);

  double val = 0.0;
  for(Index k=0;k<gi.r;k++)
    val += dx(k, k) + dy(k, k);
  return 0.5/gi.C_P*val;
}

double magnetic_energy(const mat& KA, const grid_info<2>& gi, const blas_ops& blas) {
  if(gi.C_A < 1e-10)
    return 0.0;
  else
    return gi.C_P/gi.C_A*electric_energy(KA, gi, blas);
}

struct vector_potential {

  void operator()(const lr2<double>& f, mat& KA, mat& VA) {
    // compute the basis of -<v V_j^f>_v
    integrate_mulv_v(f.V, VA, gi);
    gs(VA, intVf_R, ip_z);

    blas.matmul_transb(f.S, intVf_R, Stmp);

    blas.matmul(f.X, Stmp, Krhs);


    if(fft == nullptr)
      fft = make_unique_ptr<fft2d<2>>(gi.N_xx, Krhs, KdtAhat);

    fft->forward(Krhs, KdtAhat);

    double ncxx = 1.0/double(gi.N_xx[0]*gi.N_xx[1]);
    componentwise_mat_fourier_omp(gi.r, gi.N_xx, [this, ncxx](Index idx, mind<2> i, Index r) {
      Index mult_j = freq(i[1], gi.N_xx[1]);
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[1]-gi.lim_xx[0])*i[0]);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[3]-gi.lim_xx[2])*mult_j);
          
      KdtAhat(idx, r) *= (i[0]==0 && mult_j==0) ? 0.0 : gi.C_A/(pow(lambdax,2) + pow(lambday,2))*ncxx;
    });

    fft->backward(KdtAhat, KA);
  }

  vector_potential(const grid_info<2>& _gi, const blas_ops& _blas) :gi(_gi), blas(_blas), gs(&_blas) {
    intvVf.resize({gi.N_zv[0], gi.r});
    intVf_R.resize({gi.r, gi.r});
    Krhs.resize({gi.dxx_mult, gi.r});
    KdtAhat.resize({gi.dxxh_mult, gi.r});
    Stmp.resize({gi.r,gi.r});
    ip_z = inner_product_from_const_weight(gi.h_zv[0], gi.N_zv[0]);
  }

private:
  grid_info<2> gi;
  const blas_ops& blas;
  gram_schmidt gs;
  mat intvVf, intVf_R;
  mat Krhs, Kphi, Stmp;
  cmat KdtAhat;
  std::unique_ptr<fft2d<2>> fft;
  std::function<double(double*,double*)> ip_z;
};

double max_norm_estimate(const mat& K, const mat& V, const grid_info<2>& gi, const blas_ops& blas) {
  mat Knorm({1,gi.r}), Vnorm({1,gi.r});

  for(Index k=0;k<gi.r;k++) {
    double max_K = 0.0;
    for(Index i=0;i<K.shape()[0];i++) {
      max_K = max(max_K, abs(K(i, k)));
    }
    Knorm(0, k) = max_K;
  }
  
  for(Index k=0;k<gi.r;k++) {
    double max_V = 0.0;
    for(Index i=0;i<V.shape()[0];i++) {
      max_V = max(max_V, abs(V(i, k)));
    }
    Vnorm(0, k) = max_V;
  }

  mat result({1,1});
  blas.matmul_transb(Knorm, Vnorm, result);
  return result(0,0);
}


void integrate_mulvsq(const mat& V, mat& intV, const grid_info<2>& gi) {
  for(Index r=0;r<gi.r;r++) {
    for(Index iz=0;iz<gi.N_zv[0];iz++) {
      double s = 0.0;
      for(Index iv=0;iv<gi.N_zv[1];iv++)
        s += pow(gi.v(1, iv),2)*V(gi.lin_idx_v({iz,iv}), r)*gi.h_zv[1];
      intV(iz, r) = s;
    }
  }
}


struct dtA_iterative_solver {

  void compute_rhs(const lr2<double>& f, const lr2<double>& E, const lr2<double>& mrho) {
    gt::start("dtA_iterative_rhs");

    // \partial_z \int v^2 f dv
    int_vsq_z.X = f.X;
    int_vsq_z.S = f.S;
    integrate_mulvsq(f.V, int_vsq_z.V, gi);
    deriv_z(int_vsq_z.V, int_vsq_z.V, gi);

    // E (1-\rho)
    gt::start("dtA_iterative_mul");
    lr_mul(E, mrho, prod, ip_xx, ip_z, blas);
    gt::stop("dtA_iterative_mul");

    // add them together and truncate
    gt::start("dtA_iterative_add");
    lr_add(gi.C_A, int_vsq_z, -gi.C_A/gi.M_e, prod, large, ip_xx, ip_z, blas);
    gt::stop("dtA_iterative_add");
    lr_truncate(large, rhs, blas);
    
    gt::stop("dtA_iterative_rhs");
  }

  void apply_lhs(lr2<double>& dtA, const lr2<double>& mrho) {
    gt::start("dtA_iterative_lhs");

    // (1-rho) \partial_t A
    gt::start("dtA_iterative_mul");
    lr_mul(mrho, dtA, prod, ip_xx, ip_z, blas);
    gt::stop("dtA_iterative_mul");

    // (-\partial_xx - \partial_yy) A
    if(fft == nullptr)
      fft = make_unique_ptr<fft2d<2>>(gi.N_xx, dtA.X, Xhat);

    fft->forward(dtA.X, Xhat);

    double ncxx = 1.0/double(gi.N_xx[0]*gi.N_xx[1]);
    componentwise_mat_fourier_omp(gi.r, gi.N_xx, [this, ncxx](Index idx, mind<2> i, Index r) {
      Index mult_j = freq(i[1], gi.N_xx[1]);
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[1]-gi.lim_xx[0])*i[0]);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(gi.lim_xx[3]-gi.lim_xx[2])*mult_j);
          
      Xhat(idx, r) *= -(pow(lambdax,2) + pow(lambday,2))*ncxx;
    });

    fft->backward(Xhat, dtA.X);

    gt::start("dtA_iterative_add");
    lr_add(1.0, dtA, gi.C_A/gi.M_e, prod, large, ip_xx, ip_z, blas);
    gt::stop("dtA_iterative_add");
    lr_truncate(large, dtA, blas);
    
    gt::stop("dtA_iterative_lhs");
  }

  void operator()(lr2<double>& dtA, const lr2<double>& f, const lr2<double>& E, const lr2<double>& mrho, lr2<double>* _rhs=nullptr) {
    if(_rhs == nullptr)
      compute_rhs(f, E, mrho);
    else
      rhs = *_rhs;

    Ap = dtA;
    apply_lhs(Ap, mrho);

    lr_add(1.0, rhs, -1.0, Ap, medium, ip_xx, ip_z, blas);
    lr_truncate(medium, r, blas);
    p = r;
    double r_norm_sq = lr_norm_sq(r, blas);

    bool iteration_finished = false;
    for(Index it=0;it<200;it++) {
      Ap = p;
      apply_lhs(Ap, mrho);

      double alpha = r_norm_sq / lr_inner_product(p, Ap, gi.h_xx[0]*gi.h_xx[1]*gi.h_zv[0], blas);

      //cout << "dtA2: " << lr_norm_sq(dtA, blas) << endl;
      lr_add(1.0, dtA, alpha, p, medium, ip_xx, ip_z, blas);
      lr_truncate(medium, dtA, blas);
      //cout << "dtA3: " << lr_norm_sq(dtA, blas) << endl;


      lr_add(1.0, r, -alpha, Ap, medium, ip_xx, ip_z, blas);
      lr_truncate(medium, r, blas);

      double r_norm_sq_new = lr_norm_sq(r, blas);
      if(sqrt(r_norm_sq_new) < 1e-12) {
        iteration_finished = true;
        break;
      }

      double beta = r_norm_sq_new/r_norm_sq;
      lr_add(1.0, r, beta, p, medium, ip_xx, ip_z, blas);
      lr_truncate(medium, p, blas);

      r_norm_sq = r_norm_sq_new;
    }

    if(!iteration_finished)
      cout << "WARNING: dtA_iterative_solver did not converge within 200 iterations" << endl;
  }


  dtA_iterative_solver(const grid_info<2>& _gi, const blas_ops& _blas)
    : gi(_gi), blas(_blas), int_vsq_z(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}), prod(_gi.r*_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),
      medium(2*_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}), large(_gi.r*_gi.r + _gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),
      rhs(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),
      r(_gi.r, {gi.dxx_mult, gi.N_zv[0]}), p(_gi.r, {gi.dxx_mult, gi.N_zv[0]}), Ap(_gi.r, {gi.dxx_mult, gi.N_zv[0]}) {
    Xhat.resize({gi.dxxh_mult, gi.r});
    ip_xx = inner_product_from_const_weight(gi.h_xx[0]*gi.h_xx[1], gi.dxx_mult);
    ip_z = inner_product_from_const_weight(gi.h_zv[0], gi.N_zv[0]);
  }

  grid_info<2> gi;
  const blas_ops& blas;
  std::function<double(double*,double*)> ip_xx, ip_z;
  lr2<double> int_vsq_z, prod, medium, large, rhs, r, p, Ap;
  cmat Xhat;
  std::unique_ptr<fft2d<2>> fft;
};





struct timestepper {

  timestepper() : __Kphi(nullptr), __Vphi(nullptr) {}
  virtual ~timestepper() {}

  virtual void operator()(double tau, lr2<double>& f, double* ee=nullptr, double* ee_max=nullptr) = 0;

  void set_Kphi(mat* Kphi) {
    __Kphi = Kphi;
  }
  
  void set_Vphi(mat* Vphi) {
    __Vphi = Vphi;
  }

  void set_dtA(lr2<double>* dtA) {
    __dtA = dtA;
  }

  mat *__Kphi, *__Vphi;
  lr2<double>* __dtA;
};

struct timestepper_lie : timestepper {

  void compute_E_dtA(lr2<double>& f) {
    gt::start("compute_E");
    blas.matmul(f.X,f.S,K);

    if(__Kphi == nullptr) {
      compute_phi.operator()(K, f.V, E.X, rho.V, &rho.X);
      Kphi = E.X;
      Vphi = rho.V;
    } else  {
      Kphi = *__Kphi;
      Vphi = *__Vphi;
    }

    deriv_z(rho.V, E.V, gi);
    for(Index j=0;j<gi.r;j++) {
      for(Index i=0;i<gi.r;i++) {
        rho.S(i,j) = double(i==j);
        E.S(i,j)   = double(i==j);
      }
    }
    gt::stop("compute_E");
    
    if(__dtA == nullptr) {
      gt::start("dtA_iteration");
      dtA_it.operator()(dtA, f, E, rho);
      gt::stop("dtA_iteration");
    } else {
      dtA = *__dtA;
    }
  }


  void operator()(double tau, lr2<double>& f, double* ee=nullptr, double* ee_max=nullptr) {
    gt::start("step");

    compute_E_dtA(f);

    double _ee = electric_energy(Kphi, gi, blas);
    if(ee != nullptr)
      *ee = _ee;

    if(ee_max != nullptr) {
      deriv_z(Vphi, dzVphi, gi);
      *ee_max = max_norm_estimate(Kphi, dzVphi, gi, blas); 
    }

    // K step

    C1 = ccoeff.compute_C1(f.V, blas);
    C2 = ccoeff.compute_C2(f.V, Vphi, blas);
    C3 = ccoeff.compute_C3(f.V, dtA.V, blas);
    blas.matmul(dtA.X, dtA.S, KdtA);

    gt::start("K_step");
    K_step(tau, K, Kphi, KdtA, C1, C2, C3, blas);
    gt::stop("K_step");

    gt::start("gs");

    f.X = K;
    gs(f.X, f.S, ip_xx);

    Xphi = Kphi;
    gs(Xphi, Sphi, ip_xx);

    gt::stop("gs");

    // S step

    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);
    
    gt::start("S_step");
    S_step(tau, f.S, Sphi, dtA.S, C1, C2, C3, D2, D3);
    gt::stop("S_step");

    // L step

    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    blas.matmul_transb(Vphi,Sphi,Lphi);
    e = ccoeff.compute_e(D2, Lphi);

    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);
    blas.matmul_transb(dtA.V,dtA.S,LdtA);
    eA = ccoeff.compute_eA(D3, LdtA);

    gt::start("L_step");
    blas.matmul_transb(f.V,f.S,L);
    L_step(tau, L, e, eA);
    gt::stop("L_step");

    gt::start("gs");

    f.V = L;
    gs(f.V, f.S, ip_zv);
    transpose_inplace(f.S);

    gt::stop("gs");

    gt::stop("step");
  }

  timestepper_lie(const grid_info<2>& _gi, const blas_ops& _blas, std::function<double(double*,double*)> _ip_xx, std::function<double(double*,double*)> _ip_zv) : gi(_gi), blas(_blas), ccoeff(_gi), L_step(_gi, _blas), K_step(_gi, _blas), S_step(_gi, _blas), gs(&_blas), compute_phi(_gi, _blas), dtA_it(_gi, _blas), ip_xx(_ip_xx), ip_zv(_ip_zv), rho(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}), E(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),  dtA(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}) {
    D2.resize({gi.r,gi.r,gi.r});
    D3.resize({gi.r,gi.r,gi.r});
    e.resize({gi.N_zv[0], gi.r, gi.r});
    eA.resize({gi.N_zv[0], gi.r, gi.r});
    C1.resize({gi.r, gi.r});
    C2.resize({gi.r,gi.r,gi.r});
    C3.resize({gi.r,gi.r,gi.r});

    Lphi.resize({gi.N_zv[0], gi.r});
    LdtA.resize({gi.N_zv[0], gi.r});
    Sphi.resize({gi.r, gi.r});

    K.resize({gi.dxx_mult, gi.r});
    L.resize({gi.dzv_mult, gi.r});

    Kphi.resize({gi.dxx_mult, gi.r});
    dzVphi.resize({gi.N_zv[0], gi.r});
    KdtA.resize({gi.dxx_mult, gi.r});
    Xphi.resize({gi.dxx_mult, gi.r});
    Vphi.resize({gi.N_zv[0], gi.r});
  }

private:
  grid_info<2> gi;
  const blas_ops& blas;
  compute_coeff ccoeff;
  PS_L_step L_step;
  PS_K_step K_step;
  PS_S_step S_step;
  gram_schmidt gs;
  scalar_potential compute_phi;
  dtA_iterative_solver dtA_it;
  mat C1;
  ten3 D2, e, eA, C2, C3, D3;
  mat Lphi, Sphi, K, L, Kphi, Xphi, Vphi, LdtA, KdtA, dzVphi;
  std::function<double(double*,double*)> ip_xx, ip_zv;
  lr2<double> rho, E, dtA;
};


struct timestepper_unconventional: timestepper {

  void compute_E_dtA(lr2<double>& f) {
    gt::start("compute_E");
    blas.matmul(f.X,f.S,K);

    if(__Kphi == nullptr) {
      compute_phi.operator()(K, f.V, E.X, rho.V, &rho.X);
      Kphi = E.X;
      Vphi = rho.V;
    } else  {
      Kphi = *__Kphi;
      Vphi = *__Vphi;
    }

    deriv_z(rho.V, E.V, gi);
    for(Index j=0;j<gi.r;j++) {
      for(Index i=0;i<gi.r;i++) {
        rho.S(i,j) = double(i==j);
        E.S(i,j)   = double(i==j);
      }
    }
    gt::stop("compute_E");
    
    if(__dtA == nullptr) {
      gt::start("dtA_iteration");
      dtA_it.operator()(dtA, f, E, rho);
      gt::stop("dtA_iteration");
    } else {
      dtA = *__dtA;
    }
  }


  void operator()(double tau, lr2<double>& f, double* ee=nullptr, double* ee_max=nullptr) {
    gt::start("step");

    compute_E_dtA(f);

    double _ee = electric_energy(Kphi, gi, blas);
    if(ee != nullptr)
      *ee = _ee;

    if(ee_max != nullptr) {
      deriv_z(Vphi, dzVphi, gi);
      *ee_max = max_norm_estimate(Kphi, dzVphi, gi, blas); 
    }

    // backup initial value
    X0 = f.X;
    V0 = f.V;

    // compute the coefficients
    C1 = ccoeff.compute_C1(f.V, blas);
    C2 = ccoeff.compute_C2(f.V, Vphi, blas);
    C3 = ccoeff.compute_C3(f.V, dtA.V, blas);

    Xphi = Kphi;
    gt::start("gs");
    gs(Xphi, Sphi, ip_xx);
    gt::stop("gs");

    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);

    blas.matmul_transb(Vphi,Sphi,Lphi);
    e = ccoeff.compute_e(D2, Lphi);
    
    blas.matmul_transb(dtA.V,dtA.S,LdtA);
    eA = ccoeff.compute_eA(D3, LdtA);

    // K step
    blas.matmul(dtA.X, dtA.S, KdtA);
    gt::start("K_step");
    K_step(tau, K, Kphi, KdtA, C1, C2, C3, blas);
    gt::stop("K_step");

    // L step
    gt::start("L_step");
    blas.matmul_transb(f.V,f.S,L);
    L_step(tau, L, e, eA);
    gt::stop("L_step");

    gt::start("gs");
    f.X = K;
    gs(f.X, unused, ip_xx);
    f.V = L;
    gs(f.V, unused, ip_zv);
    gt::stop("gs");

    gt::start("projection");
    
    // computer the S matrix transformed to the new basis
    mat M({gi.r,gi.r}), N({gi.r,gi.r});

    coeff(f.X, X0, gi.h_xx[0]*gi.h_xx[1], M, blas);
    coeff(f.V, V0, gi.h_zv[0]*gi.h_zv[1], N, blas);

    mat Stmp({gi.r,gi.r});
    blas.matmul(M, f.S, Stmp);
    blas.matmul_transb(Stmp, N, f.S);
    gt::stop("projection");

    // recompute the required coefficients (with the new basis functions)
    C1 = ccoeff.compute_C1(f.V, blas);
    C2 = ccoeff.compute_C2(f.V, Vphi, blas);
    C3 = ccoeff.compute_C3(f.V, dtA.V, blas);

    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);

    // do the S step
    gt::start("S_step");
    // Note that the S_step is implemented for the projector splitting which uses
    // the negative of the rhs that we require here.
    S_step(-tau, f.S, Sphi, dtA.S, C1, C2, C3, D2, D3);
    gt::stop("S_step");

    gt::stop("step");
  }


  timestepper_unconventional(const grid_info<2>& _gi, const blas_ops& _blas, std::function<double(double*,double*)> _ip_xx, std::function<double(double*,double*)> _ip_zv) : gi(_gi), blas(_blas), ccoeff(_gi), L_step(_gi, _blas), K_step(_gi, _blas), S_step(_gi, _blas), gs(&_blas), compute_phi(_gi, _blas), dtA_it(_gi, _blas), ip_xx(_ip_xx), ip_zv(_ip_zv), rho(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}), E(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),  dtA(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}) {
    D2.resize({gi.r,gi.r,gi.r});
    D3.resize({gi.r,gi.r,gi.r});
    e.resize({gi.N_zv[0], gi.r, gi.r});
    eA.resize({gi.N_zv[0], gi.r, gi.r});
    C1.resize({gi.r, gi.r});
    C2.resize({gi.r,gi.r,gi.r});
    C3.resize({gi.r,gi.r,gi.r});

    Lphi.resize({gi.N_zv[0], gi.r});
    LdtA.resize({gi.N_zv[0], gi.r});
    Sphi.resize({gi.r, gi.r});
    unused.resize({gi.r, gi.r});

    K.resize({gi.dxx_mult, gi.r});
    L.resize({gi.dzv_mult, gi.r});

    Kphi.resize({gi.dxx_mult, gi.r});
    dzVphi.resize({gi.N_zv[0], gi.r});
    KdtA.resize({gi.dxx_mult, gi.r});
    Xphi.resize({gi.dxx_mult, gi.r});
    Vphi.resize({gi.N_zv[0], gi.r});

    X0.resize({gi.dxx_mult, gi.r});
    V0.resize({gi.dzv_mult, gi.r});
  }

private:
  grid_info<2> gi;
  const blas_ops& blas;
  compute_coeff ccoeff;
  PS_L_step L_step;
  PS_K_step K_step;
  PS_S_step S_step;
  gram_schmidt gs;
  scalar_potential compute_phi;
  dtA_iterative_solver dtA_it;
  mat C1;
  ten3 D2, e, eA, C2, C3, D3;
  mat Lphi, Sphi, unused, K, L, Kphi, Xphi, Vphi, LdtA, KdtA, dzVphi, X0, V0;
  std::function<double(double*,double*)> ip_xx, ip_zv;
  lr2<double> rho, E, dtA;

};


struct timestepper_augmented_unconventional: timestepper {

  void compute_E_dtA(lr2<double>& f) {
    gt::start("compute_E");
    blas.matmul(f.X,f.S,K);

    if(__Kphi == nullptr) {
      compute_phi.operator()(K, f.V, E.X, rho.V, &rho.X);
      Kphi = E.X;
      Vphi = rho.V;
    } else  {
      Kphi = *__Kphi;
      Vphi = *__Vphi;
    }

    deriv_z(rho.V, E.V, gi);
    for(Index j=0;j<gi.r;j++) {
      for(Index i=0;i<gi.r;i++) {
        rho.S(i,j) = double(i==j);
        E.S(i,j)   = double(i==j);
      }
    }
    gt::stop("compute_E");
    
    if(__dtA == nullptr) {
      gt::start("dtA_iteration");
      dtA_it.operator()(dtA, f, E, rho);
      gt::stop("dtA_iteration");
    } else {
      dtA = *__dtA;
    }
  }

  void operator()(double tau, lr2<double>& f, double* ee=nullptr, double* ee_max=nullptr) {
    gt::start("step");

    compute_E_dtA(f);

    double _ee = electric_energy(Kphi, gi, blas);
    if(ee != nullptr)
      *ee = _ee;

    if(ee_max != nullptr) {
      deriv_z(Vphi, dzVphi, gi);
      *ee_max = max_norm_estimate(Kphi, dzVphi, gi, blas); 
    }

    // backup initial value
    lr2<double> f0 = f;
    X0 = f.X;
    V0 = f.V;

    // compute the coefficients
    C1 = ccoeff.compute_C1(f.V, blas);
    C2 = ccoeff.compute_C2(f.V, Vphi, blas);
    C3 = ccoeff.compute_C3(f.V, dtA.V, blas);

    Xphi = Kphi;
    gt::start("gs");
    gs(Xphi, Sphi, ip_xx);
    gt::stop("gs");

    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);

    blas.matmul_transb(Vphi,Sphi,Lphi);
    e = ccoeff.compute_e(D2, Lphi);
    
    blas.matmul_transb(dtA.V,dtA.S,LdtA);
    eA = ccoeff.compute_eA(D3, LdtA);

    // K step
    blas.matmul(dtA.X, dtA.S, KdtA);
    gt::start("K_step");
    K_step(tau, K, Kphi, KdtA, C1, C2, C3, blas);
    gt::stop("K_step");

    // L step
    gt::start("L_step");
    blas.matmul_transb(f.V,f.S,L);
    L_step(tau, L, e, eA);
    gt::stop("L_step");

    // setup the augmented basis
    componentwise_mat_omp(gi.r, gi.N_xx, [this](Index idx, mind<2> i, Index r) {
      aug.X(idx, r) = X0(idx, r);
      aug.X(idx, gi.r+r) = K(idx, r);
    });
    componentwise_mat_omp(gi.r, gi.N_zv, [this](Index idx, mind<2> i, Index r) {
      aug.V(idx, r) = V0(idx, r);
      aug.V(idx, gi.r+r) = L(idx, r);
    });

    gt::start("gs");
    gs(aug.X, unused, ip_xx);
    gs(aug.V, unused, ip_zv);
    gt::stop("gs");

    // We do not need to compute the new S matrix (with N and M) since we already
    // know how S is expressed in the new basis.
    for(Index i=0;i<2*gi.r;i++)
      for(Index j=0;j<2*gi.r;j++)
        aug.S(i,j) = (i<gi.r && j<gi.r) ? f.S(i,j) : 0.0;
    

    // recompute the required coefficients (with the new basis functions)
    aug_C1 = aug_ccoeff.compute_C1(aug.V, blas);
    aug_C2 = aug_ccoeff.compute_C2(aug.V, Vphi, blas);
    aug_C3 = aug_ccoeff.compute_C3(aug.V, dtA.V, blas);

    aug_D2 = aug_ccoeff.compute_D2(aug.X, Xphi, blas);
    aug_D3 = aug_ccoeff.compute_D3(aug.X, dtA.X, blas);

    // do the S step
    gt::start("S_step");
    // Note that the S_step is implemented for the projector splitting which uses
    // the negative of the rhs compared to the projector splitting.
    S_step_aug(-tau, aug.S, Sphi, dtA.S, aug_C1, aug_C2, aug_C3, aug_D2, aug_D3);
    gt::stop("S_step");

    gt::start("truncation");
    lr_truncate(aug, f, blas);
    gt::stop("truncation");
    gt::stop("step");
  }


  timestepper_augmented_unconventional(const grid_info<2>& _gi, const blas_ops& _blas, std::function<double(double*,double*)> _ip_xx, std::function<double(double*,double*)> _ip_zv) : gi(_gi), blas(_blas), ccoeff(_gi), aug_ccoeff(modify_r(_gi, 2*_gi.r), gi.r), L_step(_gi, _blas), K_step(_gi, _blas), S_step(_gi, _blas), S_step_aug(modify_r(gi, 2*_gi.r), _blas), gs(&_blas), compute_phi(_gi, _blas), dtA_it(_gi, _blas), ip_xx(_ip_xx), ip_zv(_ip_zv), aug(2*_gi.r, {_gi.dxx_mult, _gi.dzv_mult}), rho(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}), E(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),  dtA(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}) {
    D2.resize({gi.r,gi.r,gi.r});
    D3.resize({gi.r,gi.r,gi.r});
    
    aug_D2.resize({2*gi.r,2*gi.r,gi.r});
    aug_D3.resize({2*gi.r,2*gi.r,gi.r});

    e.resize({gi.N_zv[0], gi.r, gi.r});
    eA.resize({gi.N_zv[0], gi.r, gi.r});

    C1.resize({gi.r, gi.r});
    C2.resize({gi.r,gi.r,gi.r});
    C3.resize({gi.r,gi.r,gi.r});
    
    aug_C1.resize({2*gi.r, 2*gi.r});
    aug_C2.resize({2*gi.r,2*gi.r,gi.r});
    aug_C3.resize({2*gi.r,2*gi.r,gi.r});

    Lphi.resize({gi.N_zv[0], gi.r});
    LdtA.resize({gi.N_zv[0], gi.r});
    Sphi.resize({gi.r, gi.r});
    unused.resize({2*gi.r, 2*gi.r});

    K.resize({gi.dxx_mult, gi.r});
    L.resize({gi.dzv_mult, gi.r});

    Kphi.resize({gi.dxx_mult, gi.r});
    dzVphi.resize({gi.N_zv[0], gi.r});
    KdtA.resize({gi.dxx_mult, gi.r});
    Xphi.resize({gi.dxx_mult, gi.r});
    Vphi.resize({gi.N_zv[0], gi.r});

    X0.resize({gi.dxx_mult, gi.r});
    V0.resize({gi.dzv_mult, gi.r});
  }

private:
  grid_info<2> gi;
  const blas_ops& blas;
  compute_coeff ccoeff, aug_ccoeff;
  PS_L_step L_step;
  PS_K_step K_step;
  PS_S_step S_step;
  PS_S_step S_step_aug;
  gram_schmidt gs;
  scalar_potential compute_phi;
  dtA_iterative_solver dtA_it;
  mat C1, aug_C1;
  ten3 D2, e, eA, C2, C3, D3, aug_C2, aug_C3, aug_D2, aug_D3;
  mat Lphi, Sphi, unused, K, L, Kphi, Xphi, Vphi, LdtA, KdtA, dzVphi, X0, V0;
  std::function<double(double*,double*)> ip_xx, ip_zv;
  lr2<double> aug;
  lr2<double> rho, E, dtA;
};


struct timestepper_strang : timestepper {

  void compute_E_dtA(lr2<double>& f) {
    blas.matmul(f.X,f.S,K);
    compute_phi.operator()(K, f.V, E.X, rho.V, &rho.X);
    Kphi = E.X;
    Vphi = rho.V;
    deriv_z(rho.V, E.V, gi);
    for(Index j=0;j<gi.r;j++) {
      for(Index i=0;i<gi.r;i++) {
        rho.S(i,j) = double(i==j);
        E.S(i,j)   = double(i==j);
      }
    }
    
    gt::start("dtA_iteration");
    dtA_it.operator()(dtA, f, E, rho);
    gt::stop("dtA_iteration");
    
    gt::start("gs");
    Xphi = Kphi;
    gs(Xphi, Sphi, ip_xx);
    gt::stop("gs");
    
    blas.matmul_transb(Vphi,Sphi,Lphi);
    blas.matmul_transb(dtA.V,dtA.S,LdtA);
    
    blas.matmul(dtA.X, dtA.S, KdtA);
  }

  void step_K(double tau, lr2<double>& f) {
    // K step (0.5*tau)
    C1 = ccoeff.compute_C1(f.V, blas);
    C2 = ccoeff.compute_C2(f.V, Vphi, blas);
    C3 = ccoeff.compute_C3(f.V, dtA.V, blas);

    gt::start("K_step");
    blas.matmul(f.X,f.S,K);
    K_step(tau, K, Kphi, KdtA, C1, C2, C3, blas);
    gt::stop("K_step");

    gt::start("gs");
    f.X = K;
    gs(f.X, f.S, ip_xx);
    gt::stop("gs");
  }

  void step_S(double tau, lr2<double>& f) {
    // S step (0.5*tau)
    C1 = ccoeff.compute_C1(f.V, blas);
    C2 = ccoeff.compute_C2(f.V, Vphi, blas);
    C3 = ccoeff.compute_C3(f.V, dtA.V, blas);
    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);

    gt::start("S_step");
    S_step(tau, f.S, Sphi, dtA.S, C1, C2, C3, D2, D3);
    gt::stop("S_step");
  }

  void step_L(double tau, lr2<double>& f) {
    // L step (0.5*tau)
    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);
    e = ccoeff.compute_e(D2, Lphi);
    eA = ccoeff.compute_eA(D3, LdtA);

    gt::start("L_step");
    blas.matmul_transb(f.V,f.S,L);
    L_step(tau, L, e, eA);
    gt::stop("L_step");

    gt::start("gs");
    f.V = L;
    gs(f.V, f.S, ip_zv);
    transpose_inplace(f.S);
    gt::stop("gs");

  }

  void operator()(double tau, lr2<double>& f, double* ee=nullptr, double* ee_max=nullptr) {
    gt::start("step");

    // compute electric and magnetic field
    compute_E_dtA(f);

    double _ee = electric_energy(Kphi, gi, blas);
    if(ee != nullptr)
      *ee = _ee;

    if(ee_max != nullptr) {
      deriv_z(Vphi, dzVphi, gi);
      *ee_max = max_norm_estimate(Kphi, dzVphi, gi, blas); 
    }

    ftmp = f;
    step_L(0.5*tau, ftmp);
    step_S(0.5*tau, ftmp);
    step_K(0.5*tau, ftmp);

    compute_E_dtA(ftmp);

    step_L(0.5*tau, f);
    step_S(0.5*tau, f);
    step_K(tau, f);
    step_S(0.5*tau, f);
    step_L(0.5*tau, f);
    
    gt::stop("step");
  }

  timestepper_strang(const grid_info<2>& _gi, const blas_ops& _blas, std::function<double(double*,double*)> _ip_xx, std::function<double(double*,double*)> _ip_zv) : gi(_gi), blas(_blas), ccoeff(_gi), L_step(_gi, _blas), K_step(_gi, _blas), S_step(_gi, _blas), gs(&_blas), compute_phi(_gi, _blas), dtA_it(_gi, _blas), ip_xx(_ip_xx), ip_zv(_ip_zv), rho(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}), E(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),  dtA(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}), ftmp(_gi.r, {_gi.dxx_mult, _gi.dzv_mult}) {
    D2.resize({gi.r,gi.r,gi.r});
    D3.resize({gi.r,gi.r,gi.r});
    e.resize({gi.N_zv[0], gi.r, gi.r});
    eA.resize({gi.N_zv[0], gi.r, gi.r});
    C1.resize({gi.r, gi.r});
    C2.resize({gi.r,gi.r,gi.r});
    C3.resize({gi.r,gi.r,gi.r});

    Lphi.resize({gi.N_zv[0], gi.r});
    LdtA.resize({gi.N_zv[0], gi.r});
    Sphi.resize({gi.r, gi.r});

    K.resize({gi.dxx_mult, gi.r});
    L.resize({gi.dzv_mult, gi.r});

    Kphi.resize({gi.dxx_mult, gi.r});
    dzVphi.resize({gi.N_zv[0], gi.r});
    KdtA.resize({gi.dxx_mult, gi.r});
    Xphi.resize({gi.dxx_mult, gi.r});
    Vphi.resize({gi.N_zv[0], gi.r});
  }

private:
  grid_info<2> gi;
  const blas_ops& blas;
  compute_coeff ccoeff;
  PS_L_step L_step;
  PS_K_step K_step;
  PS_S_step S_step;
  gram_schmidt gs;
  scalar_potential compute_phi;
  dtA_iterative_solver dtA_it;
  mat C1;
  ten3 D2, e, eA, C2, C3, D3;
  mat Lphi, Sphi, K, L, Kphi, Xphi, Vphi, LdtA, KdtA, dzVphi;
  std::function<double(double*,double*)> ip_xx, ip_zv;
  lr2<double> rho, E, dtA, ftmp;
};






struct rectangular_QR {
    int n, r;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr;

    rectangular_QR(int _n, int _r) : n(_n), r(_r), qr(_n,_r) {}

    rectangular_QR(Eigen::MatrixXd& A) : qr(A) {
        n = A.rows();
        r = A.cols();
    }

    void compute(const Eigen::MatrixXd& A) {
        qr.compute(A);
    }

    Eigen::MatrixXd Q() {
        // Internally Eigen stores the QR decomposition as a sequence of
        // Householder transformations. We multiply with the 'identity' to only
        // get the 'thin QR factorization'.
        // If qr.householderQ() is directly assigned to a MatrixXd object, the full matrix
        // is formed and performance suffers because of it.
        return qr.householderQ()*Eigen::MatrixXd::Identity(n, r);
    }

    Eigen::MatrixXd R() {
        Eigen::MatrixXd mr = qr.matrixQR().triangularView<Eigen::Upper>();
        return mr.block(0,0,r,r);
    }

};


struct integrator {

  void step(double tau, lr2<double>& f, double* ee, double* max_ee) {
    // do the timestep
    gt::start("timeloop_step");
    timestep->operator()(tau, f, ee, max_ee);
    gt::stop("timeloop_step");
  }
  

  double compute_ee(lr2<double>& f) {
    blas.matmul(f.X,f.S,K);
    compute_phi(K, f.V, Kphi, Vphi);
    return electric_energy(Kphi, gi, blas);
  }

  lr2<double> run() {

    blas.matmul(f.X,f.S,K);
    double ke0 = kinetic_energy(K, f.V, gi, blas);

    ofstream fs_evolution("evolution.data");
    fs_evolution << "#PARAMETERS: final_time=" << final_time << " deltat=" << tau << " r=" << gi.r << " n=(" << gi.N_xx[0] << "," << gi.N_xx[1] << "," << gi.N_zv[0] << "," << gi.N_zv[1] << ") lim_xx=(" << gi.lim_xx[0] << "," << gi.lim_xx[1] << "," << gi.lim_xx[2] << "," << gi.lim_xx[3] << ") lim_zv=(" << gi.lim_zv[0] << "," << gi.lim_zv[1] << "," << gi.lim_zv[2] << "," << gi.lim_zv[3] << ") M_e=" << gi.M_e << " C_P=" << gi.C_P << " C_A=" << gi.C_A << endl;
    fs_evolution.precision(16);

    double t = 0.0;
    bool final_step = false;
    while(t<final_time && !final_step) {
        gt::start("timestep");

        if(final_time - t < tau) {
          tau = final_time - t;
          final_step = true;
        }

        gt::start("timeloop_diagnostic");
        blas.matmul(f.X,f.S,K);
        double ke = kinetic_energy(K, f.V, gi, blas);
        double m = mass(K, f.V, gi, blas);
        double mom = momentum(K, f.V, gi, blas);
        compute_A(f, AK0, A0.V);
        gt::stop("timeloop_diagnostic");

        double ee, max_ee;
        double tau_new=tau;
        if(!time_adaptive) {
          step(tau, f, &ee, &max_ee);
        } else {
          ftmp2 = f;

          ftmp = f;
          double ee1, max_ee1;
          step(tau, ftmp, &ee1, &max_ee1);
          double ee_tau = compute_ee(ftmp);
          compute_A(ftmp, A0.X, A0.V);
          double me_tau = magnetic_energy(A0.X, gi, blas);
          
          step(0.5*tau, f, &ee, &max_ee);
          step(0.5*tau, f, &ee1, &max_ee);
          double ee_tauhalf = compute_ee(f);
          compute_A(f, A0.X, A0.V);
          double me_tauhalf = magnetic_energy(A0.X, gi, blas);

          double err = sqrt(pow(ee_tau-ee_tauhalf,2)+pow(me_tau-me_tauhalf,2))/double(pow(2,order+1)-1); 
          double err_rel = err/sqrt(pow(ee_tauhalf,2)+pow(me_tauhalf,2))/time_tol/tau; // relative error per unit time
          if(err_rel < 1.0) {
            // accept the step and adjust tau
            tau_new = tau*pow(0.7/err_rel, 1.0/(double(order)+1.0));
          } else {
            // reject the step
            f = ftmp2;
            tau = std::min(tau*pow(0.7/err_rel, 0.5), 0.5*tau);
            cout << "step rejected " << tau << endl;
            continue;
          }
        }

        gt::start("timeloop_diagnostic");
        double me = magnetic_energy(AK0, gi, blas);
        blas.matmul(dtA.X, dtA.S, AK0);
        double max_dtA = max_norm_estimate(AK0, dtA.V, gi, blas);
        fs_evolution << t << "\t" << ee << "\t" << me << "\t" << ke << "\t" << ke0 << "\t" << m << "\t" << mom << "\t" << max_ee<< "\t" << max_dtA << " " << tau << endl;
        cout.flush();
        gt::stop("timeloop_diagnostic");

        gt::stop("timestep");
        cout << "\r" << std::setw(30) << "";
        cout << "\rt=" << t << " time: " << gt::average("timestep");

        t += tau;
        if(time_adaptive)
          tau = tau_new;

    }
        
    cout << "\r" << std::setw(30) << "" << "\r";

    ofstream fs_timing("timing.data");
    fs_timing << endl << gt::sorted_output() << endl;

    return f;
  }

  integrator(string method, double _final_time, double _tau, bool _time_adaptive, double _time_tol, const grid_info<2>& _gi, vector<const double*> X0, vector<const double*> V0, mat* __Kphi=nullptr, mat* __Vphi=nullptr, lr2<double>* ___dtA=nullptr)
    :  final_time(_final_time), tau(_tau), time_adaptive(_time_adaptive), time_tol(_time_tol), gi(_gi), gs(&blas),
      f(_gi.r, {_gi.dxx_mult, _gi.dzv_mult}),
      ftmp(_gi.r, {_gi.dxx_mult, _gi.dzv_mult}),
      ftmp2(_gi.r, {_gi.dxx_mult, _gi.dzv_mult}),
      dtA2(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),
      dtA_large(_gi.r*3, {_gi.dxx_mult, _gi.N_zv[0]}),
      dtA_medium(_gi.r*2, {_gi.dxx_mult, _gi.N_zv[0]}),
      A0(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),
      A1(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),
      dtA(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),
      E(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),
      rho(_gi.r, {_gi.dxx_mult, _gi.N_zv[0]}),
      AK0({_gi.dxx_mult, _gi.r}),
      K({_gi.dxx_mult, _gi.r}),
      Kphi({_gi.dxx_mult, _gi.r}),
      Vphi({gi.N_zv[0], _gi.r}),
      compute_A(_gi, blas),
      dtA_it(_gi, blas),
      compute_phi(_gi, blas),
      __dtA(___dtA) {
    
    ip_xx = inner_product_from_const_weight(gi.h_xx[0]*gi.h_xx[1], gi.dxx_mult);
    ip_zv = inner_product_from_const_weight(gi.h_zv[0]*gi.h_zv[1], gi.dzv_mult);
    
    initialize(f, X0, V0, ip_xx, ip_zv, blas);

    order = 1;
    if(method == "lie") {
      timestep = make_unique_ptr<timestepper_lie>(gi, blas, ip_xx, ip_zv);
    } else if(method == "unconventional") {
      timestep = make_unique_ptr<timestepper_unconventional>(gi, blas, ip_xx, ip_zv);
    } else if(method == "augmented") {
      timestep = make_unique_ptr<timestepper_augmented_unconventional>(gi, blas, ip_xx, ip_zv);
    } else if(method == "strang") {
      timestep = make_unique_ptr<timestepper_strang>(gi, blas, ip_xx, ip_zv);
      order = 2;
    } else {
      cout << "ERROR: the method " << method << " is not valid for integration" << endl;
      exit(1);
    }

    timestep->set_Kphi(__Kphi);
    timestep->set_Vphi(__Vphi);
    timestep->set_dtA(__dtA);

    // initialize dtA by 0
    ip_z = inner_product_from_const_weight(gi.h_zv[0], gi.N_zv[0]);
    initialize(dtA, vector<const double*>(), vector<const double*>(), ip_xx, ip_z, blas);
  }

  double final_time, tau;
  bool time_adaptive;
  double time_tol;
  blas_ops blas;
  grid_info<2> gi;
  gram_schmidt gs;
  lr2<double> f, ftmp, ftmp2, dtA2, dtA_large, dtA_medium, A0, A1, dtA, E, rho;
  mat AK0, K, Kphi, Vphi;
  std::function<double(double*,double*)> ip_xx, ip_zv, ip_z;
  std::unique_ptr<timestepper> timestep;
  Index order;
  vector_potential compute_A;
  dtA_iterative_solver dtA_it;
  scalar_potential compute_phi;
  lr2<double>* __dtA;
};