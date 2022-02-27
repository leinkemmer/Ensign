#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/timer.hpp>
#include <generic/fft.hpp>

// TODO: using lapack
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

template<size_t d> using mind = array<Index,d>;
template<size_t d> using mfp  = array<double,d>;
using mat  = multi_array<double,2>;
using cmat  = multi_array<complex<double>,2>;
using ten3  = multi_array<double,3>;
using vec  = multi_array<double,1>;

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
  mind<d>  N_xx, N_zv;
  mfp<2*d> lim_xx, lim_zv;
  mfp<d>   h_xx, h_zv;
  Index dxx_mult, dzv_mult, dxxh_mult;
  double M_e, C_P, C_A;

  bool debug_adv_z, debug_adv_v, debug_adv_vA;

  grid_info(Index _r, mind<d> _N_xx, mind<d> _N_zv, mfp<2*d> _lim_xx, mfp<2*d> _lim_zv, double _M_e, double _C_P, double _C_A)
    : r(_r), N_xx(_N_xx), N_zv(_N_zv), lim_xx(_lim_xx), lim_zv(_lim_zv), M_e(_M_e), C_P(_C_P), C_A(_C_A), debug_adv_z(true), debug_adv_v(true), debug_adv_vA(true) {

    // compute h_xx and h_zv
    for(int ii = 0; ii < 2; ii++){
      Index jj = 2*ii;
      h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
      h_zv[ii] = (lim_zv[jj+1]-lim_zv[jj])/ N_zv[ii];
    }
    
    dxx_mult  = N_xx[0]*N_xx[1];
    dzv_mult  = N_zv[0]*N_zv[1];
    dxxh_mult = N_xx[1]*(N_xx[0]/2 + 1);
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


// Note that using a std::function object has a non-negligible performance overhead
template<class func>
void componentwise_vec_omp(const mind<2>& N, func F) {
  /*#ifdef __OPENMP__
  #pragma omp parallel for
  #endif*/
  for(Index j = 0; j < N[1]; j++){
    for(Index i = 0; i < N[0]; i++){
      Index idx = i+j*N[0];
      F(idx, {i,j});
    }
  }
}

template<class func>
void componentwise_mat_omp(Index r, const mind<2>& N,  func F) {
  /*
  #ifdef __OPENMP__
  #pragma omp parallel for collapse(2)
  #endif
  */
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
  /*
  #ifdef __OPENMP__
  #pragma omp parallel for collapse(2)
  #endif
  */
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
  if(in.shape()[0] == gi.N_zv[0] && out.shape()[0] == gi.N_zv[0]) {
    // only depends on z
    for(Index ir=0;ir<gi.r;ir++) {
      for(Index iz=0;iz<gi.N_zv[0];iz++)
        out(iz, ir) = (in((iz+1)%gi.N_zv[0], ir) - in((iz-1+gi.N_zv[0])%gi.N_zv[0], ir))/(2.0*gi.h_zv[0]);
    }
  } else if(in.shape()[0] == gi.N_zv[0] && out.shape()[0] != gi.N_zv[0]) {
    // depends on both z and v
    for(Index ir=0;ir<gi.r;ir++) {
      for(Index iv=0;iv<gi.N_zv[1];iv++) {
        for(Index iz=0;iz<gi.N_zv[0];iz++)
          out(gi.lin_idx_v({iz,iv}), ir) = (in((iz+1)%gi.N_zv[0], ir) - in((iz-1+gi.N_zv[0])%gi.N_zv[0], ir))/(2.0*gi.h_zv[0]);
      }
    }
  } else {
    // depends on both z and v
    for(Index ir=0;ir<gi.r;ir++) {
      for(Index iv=0;iv<gi.N_zv[1];iv++) {
        for(Index iz=0;iz<gi.N_zv[0];iz++)
          out(gi.lin_idx_v({iz,iv}), ir) = (in(gi.lin_idx_v({(iz+1)%gi.N_zv[0],iv}), ir) - in(gi.lin_idx_v({(iz-1+gi.N_zv[0])%gi.N_zv[0],iv}), ir))/(2.0*gi.h_zv[0]);
      }
    }
  }
}

void deriv_v(const mat& in, mat& out, const grid_info<2>& gi) {
  // depends on both z and v
  for(Index ir=0;ir<gi.r;ir++) {
    for(Index iv=0;iv<gi.N_zv[1];iv++) {
      for(Index iz=0;iz<gi.N_zv[0];iz++)
        out(gi.lin_idx_v({iz,iv}), ir) = (in(gi.lin_idx_v({iz,(iv+1)%gi.N_zv[1]}), ir) - in(gi.lin_idx_v({iz, (iv-1+gi.N_zv[1])%gi.N_zv[1]}), ir))/(2.0*gi.h_zv[1]);
    }
  }
}



void deriv_x(const mat& in, mat& out, const grid_info<2>& gi) {
  for(Index ir=0;ir<gi.r;ir++) {
    for(Index ix=0;ix<gi.N_xx[0];ix++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++)
        out(gi.lin_idx_x({ix,iy}), ir) = (in(gi.lin_idx_x({(ix+1)%gi.N_xx[0],iy}), ir) - in(gi.lin_idx_x({(ix-1+gi.N_xx[0])%gi.N_xx[0], iy}), ir))/(2.0*gi.h_xx[0]);
    }
  }
}

void deriv_y(const mat& in, mat& out, const grid_info<2>& gi) {
  for(Index ir=0;ir<gi.r;ir++) {
    for(Index ix=0;ix<gi.N_xx[0];ix++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++)
        out(gi.lin_idx_x({ix,iy}), ir) = (in(gi.lin_idx_x({ix,(iy+1)%gi.N_xx[1]}), ir) - in(gi.lin_idx_x({ix, (iy-1+gi.N_xx[1])%gi.N_xx[1]}), ir))/(2.0*gi.h_xx[1]);
    }
  }
}



struct compute_coeff {
  mat compute_C1(const mat& V, const blas_ops& blas) {
    ptw_mult_row(V,v,Vtmp1); // multiply by V
    deriv_z(V, Vtmp2, gi);

    mat C1({gi.r, gi.r});
    coeff(Vtmp1, Vtmp2, gi.h_zv[0]*gi.h_zv[1], C1, blas);
    return C1;
  }

  ten3 compute_C2(const mat& Vf, const mat& Vphi, const blas_ops& blas) {
    deriv_v(Vf, Vtmp1, gi);
    deriv_z(Vphi, Vtmp2, gi);

    ten3 C2({gi.r,gi.r,gi.r});
    coeff(Vf, Vtmp1, Vtmp2, gi.h_zv[0]*gi.h_zv[1], C2, blas);
    return C2;
  }
  
  ten3 compute_C3(const mat& Vf, const mat& VdtA, const blas_ops& blas) {
    deriv_v(Vf, Vtmp1, gi);

    // Vtmp2 has to be enlarged to the full zv space such that the coeff function
    // can be used.
    componentwise_mat_omp(gi.r, gi.N_zv, [this, &VdtA](Index idx, mind<2> i, Index r) {
      Vtmp2(idx, r) = VdtA(i[0], r);
    });

    ten3 C3({gi.r,gi.r,gi.r});
    coeff(Vf, Vtmp1, Vtmp2, gi.h_zv[0]*gi.h_zv[1], C3, blas);
    return C3;
  }
  
  ten3 compute_D2(const mat& Xf, const mat& Xphi, const blas_ops& blas) {
    ten3 D2({gi.r,gi.r,gi.r});
    coeff(Xf, Xf, Xphi, gi.h_xx[0]*gi.h_xx[1], D2, blas);
    return D2;
  }
  
  ten3 compute_D3(const mat& Xf, const mat& XdtA, const blas_ops& blas) {
    ten3 D3({gi.r,gi.r,gi.r});
    coeff(Xf, Xf, XdtA, gi.h_xx[0]*gi.h_xx[1], D3, blas);
    return D3;
  }

  ten3 compute_e(const ten3& D2, const mat& Lphi) {
    deriv_z(Lphi, dzLphi, gi);

    ten3 e({gi.N_zv[0], gi.r, gi.r});
    for(Index i=0;i<gi.r;i++) {
      for(Index k=0;k<gi.r;k++) {
        for(Index iz=0;iz<gi.N_zv[0];iz++) {
          double val = 0.0;
          for(Index m=0;m<gi.r;m++)
            val += D2(i, k, m)*dzLphi(iz, m);
          e(iz, i, k) = val;
        }
      }
    }
    return e;
  }
  
  ten3 compute_eA(const ten3& D3, const mat& LdtA) {
    ten3 eA({gi.N_zv[0], gi.r, gi.r});
    for(Index i=0;i<gi.r;i++) {
      for(Index k=0;k<gi.r;k++) {
        for(Index iz=0;iz<gi.N_zv[0];iz++) {
          double val = 0.0;
          for(Index m=0;m<gi.r;m++)
            val += D3(i, k, m)*LdtA(iz, m);
          eA(iz, i, k) = val;
        }
      }
    }
    return eA;
  }

  compute_coeff(grid_info<2> _gi) : gi(_gi) {
    Vtmp1.resize({gi.dzv_mult,gi.r});
    Vtmp2.resize({gi.dzv_mult,gi.r});
    dzLphi.resize({gi.N_zv[0], gi.r});

    v.resize({gi.dzv_mult});
    componentwise_vec_omp(gi.N_zv, [this](Index idx, mind<2> i) {
      v(idx) = gi.v(1, i[1]);
    });
  }

private:
  grid_info<2> gi;
  mat Vtmp1, Vtmp2;
  mat dzLphi;
  vec v;
};



void rk4(double tau, mat& U, std::function<void(const mat&, mat&)> rhs) {
  // TODO: this is not very efficient
  mat k1(U);
  mat k2(U);
  mat k3(U);
  mat k4(U);
  mat tmp(U);
  mat in = U;

  // k1
  rhs(in, k1);

  // k2
  tmp = k1;
  tmp *= 0.5*tau;
  tmp += in;
  rhs(tmp, k2);

  // k3
  tmp = k2;
  tmp *= 0.5*tau;
  tmp += in;
  rhs(tmp, k3);

  // k4
  tmp = k3;
  tmp *= tau;
  tmp += in;
  rhs(tmp, k4);
  
  k1 *= 1.0/6.0*tau;
  U += k1;
  k2 *= 1.0/6.0*tau*2.0; 
  U += k2;
  k3 *= 1.0/6.0*tau*2.0;
  U += k3;
  k4 *= 1.0/6.0*tau;
  U += k4;
}




struct PS_K_step {

  void operator()(double tau, mat& K, const mat& Kphi, const mat& KdtA, const mat& C1, const ten3& C2, const ten3& C3, const blas_ops& blas) {

    rk4(tau, K, [this, &C1, &C2, &C3, &Kphi, &KdtA, &blas](const mat& K, mat& Kout) {

      blas.matmul_transb(K, C1, Kout);
      Kout *= -double(gi.debug_adv_z);

      for(Index j=0;j<gi.r;j++) {
        for(Index l=0;l<gi.r;l++) {
          for(Index n=0;n<gi.r;n++) {
            for(Index i=0;i<gi.dxx_mult;i++) {
              Kout(i, j) -= double(gi.debug_adv_v)/gi.M_e*C2(j,l,n)*Kphi(i, n)*K(i, l)
                            + double(gi.debug_adv_vA)/gi.M_e*C3(j,l,n)*KdtA(i, n)*K(i, l);
            }
          }
        }
      }
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
      
      for(Index i=0;i<gi.r;i++) {
        for(Index j=0;j<gi.r;j++) {
          for(Index l=0;l<gi.r;l++) {
            for(Index n=0;n<gi.r;n++) {
              for(Index m=0;m<gi.r;m++) {
                for(Index k=0;k<gi.r;k++) {
                  Sout(i, j) += double(gi.debug_adv_v)/gi.M_e*D2(i,k,m)*Sphi(m,n)*S(k,l)*C2(j,l,n)
                               +double(gi.debug_adv_vA)/gi.M_e*D3(i,k,m)*SdtA(m,n)*S(k,l)*C3(j,l,n);
                }
              }
            }
          }
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

    if(gi.debug_adv_z) {
      // the term -v \partial_z L using a Lax-Wendroff scheme
      componentwise_mat_omp(gi.r, gi.N_zv, [this, tau, &L](Index idx, mind<2> i, Index r) {
        Index idx_p1 = gi.lin_idx_v({(i[0]+1)%gi.N_zv[0],i[1]});
        Index idx_m1 = gi.lin_idx_v({(i[0]-1+gi.N_zv[0])%gi.N_zv[0],i[1]});
        double v = gi.v(1, i[1]);
        Ltmp(idx, r) = L(idx,r) - 0.5*tau/gi.h_zv[0]*v*(L(idx_p1, r) - L(idx_m1, r))
                        +0.5*pow(tau,2)/pow(gi.h_zv[0],2)*pow(v,2)*(L(idx_p1,r)-2.0*L(idx,r)+L(idx_m1,r));
      });
    } else {
      Ltmp = L;
    }

    if(gi.debug_adv_v) {
      mat ez({gi.r,gi.r}), T({gi.r, gi.r});
      vec lambda({gi.r});
      mat M({gi.N_zv[1], gi.r}), Mout({gi.N_zv[1], gi.r});

      for(Index iz=0;iz<gi.N_zv[0];iz++) {

        for(Index k2=0;k2<gi.r;k2++) {
          for(Index k1=0;k1<gi.r;k1++) {
            ez(k1, k2) = e(iz, k1, k2) + double(gi.debug_adv_vA)*eA(iz, k1, k2);
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
        for(Index ir=0;ir<gi.r;ir++) {
          for(Index iv=0;iv<gi.N_zv[1];iv++) {
            Index iv_p1 = (iv+1)%gi.N_zv[1];
            Index iv_m1 = (iv-1+gi.N_zv[1])%gi.N_zv[1];
            Mout(iv, ir) = M(iv, ir) - 0.5*tau/gi.h_zv[1]*lambda(ir)/gi.M_e*(M(iv_p1, ir) - M(iv_m1, ir))
                                     + 0.5*pow(tau,2)/pow(gi.h_zv[1],2)*pow(lambda(ir)/gi.M_e,2)*(M(iv_p1,ir)-2.0*M(iv,ir)+M(iv_m1,ir));
          }
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
// TODO: this does not work once openmp is switched on!!!
void integrate_v(const mat& V, mat& intV, const grid_info<2>& gi) {
  for(Index k=0;k<gi.r;k++)
    for(Index i=0;i<gi.N_zv[0];i++)
      intV(i, k) = 0.0;

  componentwise_mat_omp(gi.r, gi.N_zv, [&gi, &V, &intV](Index idx, mind<2> i, Index r) {
    intV(i[0], r) += V(idx, r)*gi.h_zv[1];
  });
}

void integrate_mulv_v(const mat& V, mat& intV, const grid_info<2>& gi) {
  for(Index k=0;k<gi.r;k++)
    for(Index i=0;i<gi.N_zv[0];i++)
      intV(i, k) = 0.0;

  componentwise_mat_omp(gi.r, gi.N_zv, [&gi, &V, &intV](Index idx, mind<2> i, Index r) {
    intV(i[0], r) += -gi.v(1, i[1])*V(idx, r)*gi.h_zv[1];
  });
}

struct scalar_potential {

  void operator()(const mat& Kf, const mat& Vf, mat& Kphi, mat& Vphi) {
    // compute the basis of <V_j^f>_v
    integrate_v(Vf, intVf, gi);
    gs(intVf, intVf_R, ip_z);

    // expand 1 in that basis
    integrate(intVf, gi.h_zv[0], expansion_1, blas);

    // Construct the K for the rhs of the quasi-neutrality equation
    componentwise_mat_omp(gi.r, gi.N_xx, [this, &Kf](Index idx, mind<2> i, Index j) {
      double val = 0.0;
      for(Index k=0;k<gi.r;k++)
        val += Kf(idx, k)*intVf_R(j, k);
      Krhs(idx, j) =  gi.C_P*(expansion_1(j) - val);
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
          
      // TODO: why is there a minus here? otherwise it does not work
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

double kinetic_energy(const mat& K, const mat& V, const grid_info<2>& gi, const blas_ops&blas) {
  vec intK({gi.r});
  integrate(K, gi.h_xx[0]*gi.h_xx[1], intK, blas);

  vec intvsqV({gi.r});
  mat vsqV({gi.dxx_mult, gi.r});
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
  // TODO: this is inefficient

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
          
      // TODO: why is there a minus here? otherwise it does not work
      KdtAhat(idx, r) *= (i[0]==0 && mult_j==0) ? 0.0 : -gi.C_A/(pow(lambdax,2) + pow(lambday,2))*ncxx;
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

struct timestepper {

  void operator()(double tau, lr2<double>& f, lr2<double>& dtA, double* ee=nullptr, double* ee_max=nullptr) {
    // phi 
    blas.matmul(f.X,f.S,K);

    if(__Kphi == nullptr) {
      compute_phi(K, f.V, Kphi, Vphi);
    } else {
      Kphi = *__Kphi;
      Vphi = *__Vphi;
    }

    double _ee = electric_energy(Kphi, gi, blas);
    if(ee != nullptr)
      *ee = _ee;

    if(ee_max != nullptr) {
      deriv_z(Kphi, dzKphi, gi);
      *ee_max = max_norm_estimate(dzKphi, Vphi, gi, blas); 
    }

    // K step
    C1 = ccoeff.compute_C1(f.V, blas);
    C2 = ccoeff.compute_C2(f.V, Vphi, blas);
    C3 = ccoeff.compute_C3(f.V, dtA.V, blas);
    blas.matmul(dtA.X, dtA.S, KdtA);

    K_step(tau, K, Kphi, KdtA, C1, C2, C3, blas);

    f.X = K;
    gs(f.X, f.S, ip_xx);

    Xphi = Kphi;
    gs(Xphi, Sphi, ip_xx);

    // S step
    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);
    S_step(tau, f.S, Sphi, dtA.S, C1, C2, C3, D2, D3);

    // L step
    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    blas.matmul_transb(Vphi,Sphi,Lphi);
    e = ccoeff.compute_e(D2, Lphi);

    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);
    blas.matmul_transb(dtA.V,dtA.S,LdtA);
    eA = ccoeff.compute_eA(D3, LdtA);

    blas.matmul_transb(f.V,f.S,L);
    L_step(tau, L, e, eA);

    f.V = L;
    gs(f.V, f.S, ip_zv);
    transpose_inplace(f.S);
  }

  timestepper(const grid_info<2>& _gi, const blas_ops& _blas, std::function<double(double*,double*)> _ip_xx, std::function<double(double*,double*)> _ip_zv) : gi(_gi), blas(_blas), ip_xx(_ip_xx), ip_zv(_ip_zv), ccoeff(_gi), L_step(_gi, _blas), K_step(_gi, _blas), S_step(_gi, _blas), gs(&_blas), compute_phi(_gi, _blas), __Kphi(nullptr), __Vphi(nullptr) {
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
    dzKphi.resize({gi.dxx_mult, gi.r});
    KdtA.resize({gi.dxx_mult, gi.r});
    Xphi.resize({gi.dxx_mult, gi.r});
    Vphi.resize({gi.N_zv[0], gi.r});
  }

  mat *__Kphi, *__Vphi;

private:
  grid_info<2> gi;
  const blas_ops& blas;
  compute_coeff ccoeff;
  PS_L_step L_step;
  PS_K_step K_step;
  PS_S_step S_step;
  gram_schmidt gs;
  scalar_potential compute_phi;
  mat C1;
  ten3 D2, e, eA, C2, C3, D3;
  mat Lphi, Sphi, K, L, Kphi, Xphi, Vphi, LdtA, KdtA, dzKphi;
  std::function<double(double*,double*)> ip_xx, ip_zv;

};


struct timestepper_strang {

  void operator()(double tau, lr2<double>& f, lr2<double>& dtA, double* ee=nullptr, double* ee_max=nullptr) {
    // phi 
    blas.matmul(f.X,f.S,K);

    if(__Kphi == nullptr) {
      compute_phi(K, f.V, Kphi, Vphi);
    } else {
      Kphi = *__Kphi;
      Vphi = *__Vphi;
    }

    double _ee = electric_energy(Kphi, gi, blas);
    if(ee != nullptr)
      *ee = _ee;

    if(ee_max != nullptr) {
      deriv_z(Kphi, dzKphi, gi);
      *ee_max = max_norm_estimate(dzKphi, Vphi, gi, blas); 
    }

    // K step 1
    C1 = ccoeff.compute_C1(f.V, blas);
    C2 = ccoeff.compute_C2(f.V, Vphi, blas);
    C3 = ccoeff.compute_C3(f.V, dtA.V, blas);
    blas.matmul(dtA.X, dtA.S, KdtA);

    K_step(0.5*tau, K, Kphi, KdtA, C1, C2, C3, blas);

    f.X = K;
    gs(f.X, f.S, ip_xx);

    Xphi = Kphi;
    gs(Xphi, Sphi, ip_xx);

    // S step 1
    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);
    S_step(0.5*tau, f.S, Sphi, dtA.S, C1, C2, C3, D2, D3);

    // L step
    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    blas.matmul_transb(Vphi,Sphi,Lphi);
    e = ccoeff.compute_e(D2, Lphi);

    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);
    blas.matmul_transb(dtA.V,dtA.S,LdtA);
    eA = ccoeff.compute_eA(D3, LdtA);

    blas.matmul_transb(f.V,f.S,L);
    L_step(tau, L, e, eA);

    f.V = L;
    gs(f.V, f.S, ip_zv);
    transpose_inplace(f.S);

    // S step 2
    D2 = ccoeff.compute_D2(f.X, Xphi, blas);
    D3 = ccoeff.compute_D3(f.X, dtA.X, blas);
    S_step(0.5*tau, f.S, Sphi, dtA.S, C1, C2, C3, D2, D3);


  }

  timestepper_strang(const grid_info<2>& _gi, const blas_ops& _blas, std::function<double(double*,double*)> _ip_xx, std::function<double(double*,double*)> _ip_zv) : gi(_gi), blas(_blas), ip_xx(_ip_xx), ip_zv(_ip_zv), ccoeff(_gi), L_step(_gi, _blas), K_step(_gi, _blas), S_step(_gi, _blas), gs(&_blas), compute_phi(_gi, _blas), __Kphi(nullptr), __Vphi(nullptr) {
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
    dzKphi.resize({gi.dxx_mult, gi.r});
    KdtA.resize({gi.dxx_mult, gi.r});
    Xphi.resize({gi.dxx_mult, gi.r});
    Vphi.resize({gi.N_zv[0], gi.r});
  }

  mat *__Kphi, *__Vphi;

private:
  grid_info<2> gi;
  const blas_ops& blas;
  compute_coeff ccoeff;
  PS_L_step L_step;
  PS_K_step K_step;
  PS_S_step S_step;
  gram_schmidt gs;
  scalar_potential compute_phi;
  mat C1;
  ten3 D2, e, eA, C2, C3, D3;
  mat Lphi, Sphi, K, L, Kphi, Xphi, Vphi, LdtA, KdtA, dzKphi;
  std::function<double(double*,double*)> ip_xx, ip_zv;

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


void add_lr(double alpha, const lr2<double>& A, double beta, const lr2<double>& B, lr2<double>& C, const blas_ops& blas) {
  using namespace Eigen;

  Index r = A.S.shape()[0];
  Index nx = A.X.shape()[0];
  Index nv = A.V.shape()[0];

  MatrixXd S_large({2*r, 2*r});
  S_large.setZero();
  for(Index j=0;j<r;j++) {
    for(Index i=0;i<r;i++) {
      S_large(i, j) = alpha*A.S(i, j);
      S_large(r+i,r+j) = beta*B.S(i, j);
    }
  }

  // X_large
  MatrixXd X_large(nx, 2*r);
  for(Index k=0;k<r;k++) {
    for(Index i=0;i<nx;i++) {
      X_large(i, k) = A.X(i, k);
      X_large(i, k+r) = B.X(i, k);
    }
  }

  // V_large 
  MatrixXd V_large(nv, 2*r);
  for(Index k=0;k<r;k++) {
    for(Index i=0;i<nv;i++) {
      V_large(i, k) = A.V(i, k);
      V_large(i, k+r) = B.V(i, k);
    }
  }

  rectangular_QR qrX(nx, 2*r);
  qrX.compute(X_large);
  S_large = qrX.R()*S_large;
  X_large = qrX.Q();

  rectangular_QR qrV(nv, 2*r);
  qrV.compute(V_large);
  S_large = S_large*qrV.R().transpose();
  V_large = qrV.Q();

/* // to check the result
  MatrixXd full = X_large*S_large*V_large.transpose();
  
  mat KA({nx, r}), A_full({nx, nv});
  blas.matmul(A.X, A.S, KA);
  blas.matmul_transb(KA, A.V, A_full);
  
  mat KB({nx, r}), B_full({nx, nv});
  blas.matmul(B.X, B.S, KB);
  blas.matmul_transb(KB, B.V, B_full);

  double err = 0.0;
  double m = 0.0;
  for(Index j=0;j<nv;j++)
    for(Index i=0;i<nx;i++) {
      err = max(err, abs(full(i,j) - alpha*A_full(i,j) - beta*B_full(i,j)) );
      m = max(m, abs(alpha*A_full(i,j) + beta*B_full(i,j)) );
    }
  cout << "add_lr: " << err << " " << m << endl;
*/

  JacobiSVD<MatrixXd> svd(S_large, ComputeThinU | ComputeThinV); 
  MatrixXd S = svd.singularValues().asDiagonal();

  // set S
  for(Index j=0;j<r;j++)
    for(Index i=0;i<r;i++)
      C.S(i, j) = S(i,j);

  X_large = X_large*svd.matrixU();
  V_large = V_large*svd.matrixV();
  //cout << X_large.rows() << " " << X_large.cols() << endl;

  for(Index k=0;k<r;k++)
    for(Index i=0;i<nx;i++)
      C.X(i, k) = X_large(i, k);
  
  for(Index k=0;k<r;k++)
    for(Index i=0;i<nv;i++)
      C.V(i, k) = V_large(i, k);

}


lr2<double> integration(double final_time, double tau, const grid_info<2>& gi, vector<const double*> X0, vector<const double*> V0, mat* __Kphi=nullptr, mat* __Vphi=nullptr) {

  blas_ops blas;
  gram_schmidt gs(&blas);

  lr2<double> f(gi.r, {gi.dxx_mult, gi.dzv_mult});
  lr2<double> ftmp(gi.r, {gi.dxx_mult, gi.dzv_mult});
  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(gi.h_xx[0]*gi.h_xx[1], gi.dxx_mult);
  std::function<double(double*,double*)> ip_zv = inner_product_from_const_weight(gi.h_zv[0]*gi.h_zv[1], gi.dzv_mult);
  initialize(f, X0, V0, ip_xx, ip_zv, blas);

  timestepper timestep(gi, blas, ip_xx, ip_zv);

  vector_potential compute_A(gi, blas);
  lr2<double> dtA2(gi.r, {gi.dxx_mult, gi.N_zv[0]});
  lr2<double> A0(gi.r, {gi.dxx_mult, gi.N_zv[0]});
  lr2<double> A1(gi.r, {gi.dxx_mult, gi.N_zv[0]});
  mat AK0({gi.dxx_mult, gi.r});
  mat K({gi.dxx_mult, gi.r});

  // initialize dtA by 0
  lr2<double> dtA(gi.r, {gi.dxx_mult, gi.N_zv[0]});
  std::function<double(double*,double*)> ip_z = inner_product_from_const_weight(gi.h_zv[0], gi.N_zv[0]);
  initialize(dtA, vector<const double*>(), vector<const double*>(), ip_xx, ip_z, blas);

/*
  for(Index j=0;j<gi.r;j++) {
  for(Index i=0;i<gi.r;i++)
  cout << dtA.S(i,j) << " ";
  cout << endl;
  }
*/
  blas.matmul(f.X,f.S,K);
  double ke0 = kinetic_energy(K, f.V, gi, blas);

  ofstream fs_evolution("evolution.data");
  fs_evolution.precision(16);
  double t = 0.0;
  Index n_steps = ceil(final_time/tau);
  for(Index ts=0;ts<n_steps;ts++) {
    if(final_time - t < tau)
      tau = final_time - t;

    blas.matmul(f.X,f.S,K);
    double ke = kinetic_energy(K, f.V, gi, blas);


    // compute \partial_t A
    ftmp = f;
    double eps = 1e-7;
    timestep(eps, ftmp, dtA);

    compute_A(f, A0.X, A0.V);
    AK0 = A0.X;
    //cout << "me: " << magnetic_energy(A0.X, gi, blas) << endl;
    gs(A0.X, A0.S, ip_xx);

    compute_A(ftmp, A1.X, A1.V);
    //cout << "me1: " << magnetic_energy(A1.X, gi, blas) << endl;
    gs(A1.X, A1.S, ip_xx);

    add_lr(1.0/eps, A1, -1.0/eps, A0, dtA2, blas);
    A0 = dtA;
    add_lr(0.99, A0, 0.01, dtA2, dtA, blas);

    //cout << "MM: " << magnetic_energy(dtA2.X, gi, blas) << endl;
/*
    for(Index j=0;j<gi.r;j++) {
      for(Index i=0;i<gi.r;i++)
        cout << dtA.S(i,j) << " ";
    cout << endl;
    }
*/
    // do the timestep
    double ee, max_ee;
    timestep(tau, f, dtA, &ee, &max_ee);

    double me = magnetic_energy(AK0, gi, blas);
    blas.matmul(dtA.X, dtA.S, AK0);
    double max_dtA = max_norm_estimate(AK0, dtA.V, gi, blas);
    fs_evolution << t << "\t" << ee << "\t" << me << "\t" << ke << "\t" << ke0 << "\t" << max_ee<< "\t" << max_dtA << endl;
    cout << "\rt=" << t;
    cout.flush();

    t += tau;
  }

  return f;
}

/*
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE( "Alfven waves", "[alfven_waves]" ) {

  blas_ops blas;
    
  double C_P = 3.0;
  double M_e = 0.1;
  Index r=3;
  mind<2> N_xx = {32, 32};
  mind<2> N_zv = {64, 64};
  mfp<4> lim_xx = {0.0,2*M_PI,0.0,2*M_PI};
  mfp<4> lim_zv = {0.0,2*M_PI,-6.0/sqrt(M_e), 6.0/sqrt(M_e)};
  grid_info<2> gi(r, N_xx, N_zv, lim_xx, lim_zv, M_e, C_P); 

  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(gi.h_xx[0]*gi.h_xx[1], gi.dxx_mult);
  std::function<double(double*,double*)> ip_zv = inner_product_from_const_weight(gi.h_zv[0]*gi.h_zv[1], gi.dzv_mult);

  vec xx1({gi.dxx_mult}), xx2({gi.dxx_mult});
  componentwise_vec_omp(gi.N_xx, [&xx1, &xx2, &gi](Index idx, array<Index,2> i) {
    mfp<2> x  = gi.x(i);
    xx1(idx) = 1.0;
    xx2(idx) = cos(x[0])*cos(x[1]);
  });

  vec vv1({gi.dzv_mult}), vv2({gi.dzv_mult});
  componentwise_vec_omp(gi.N_zv, [&vv1, &vv2, &gi](Index idx, array<Index,2> i) {
    mfp<2> zv  = gi.v(i);
    vv1(idx) = sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
    vv2(idx) = 0.5*cos(zv[0])*sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
  });

  vector<const double*> X, V;
  X.push_back(xx1.begin());
  X.push_back(xx2.begin());
  V.push_back(vv1.begin());
  V.push_back(vv2.begin());
  lr2<double> f(gi.r, {gi.dxx_mult, gi.dzv_mult});
  initialize(f, X, V, ip_xx, ip_zv, blas);

  compute_coeff cc(gi);


  SECTION("INITIALIZE") {
    mat K({gi.dxx_mult,gi.r});
    blas.matmul(f.X, f.S, K);

    mat f_full({gi.dxx_mult, gi.dzv_mult});
    blas.matmul_transb(K, f.V, f_full);

    double err = 0.0;
    for(Index jv=0;jv<gi.N_zv[1];jv++) {
      for(Index jz=0;jz<gi.N_zv[0];jz++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++) {
          for(Index ix=0;ix<gi.N_xx[0];ix++) {
            mfp<2> xy = gi.x({ix, iy});
            mfp<2> zv = gi.v({jz, jv});
            double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1],2));
            err = max(err, abs(f_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv) - expected));
          }
        }
      }
    }

    cout << "Error initial value: " << err << endl;
    REQUIRE( err < 1e-13 );
  }

  SECTION("scalar_potential"){

    mat Kphi({gi.dxx_mult,gi.r});
    mat Vphi({gi.N_zv[0],gi.r});
    
    mat K({gi.dxx_mult,gi.r});
    blas.matmul(f.X, f.S, K);
    
    scalar_potential compute_phi(gi, blas);
    compute_phi(K, f.V, Kphi, Vphi);

    mat phi({gi.dxx_mult, gi.N_zv[0]});
    blas.matmul_transb(Kphi, Vphi, phi);

    double err = 0.0;
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++) {
        for(Index ix=0;ix<gi.N_xx[0];ix++) {
          mfp<2> xy = gi.x({ix, iy});
          double z = gi.v(0, jz);
          double expected = -gi.C_P*0.25*cos(z)*cos(xy[0])*cos(xy[1]);
          err = max(err, abs(phi(ix+gi.N_xx[0]*iy,jz) - expected));
        }
      }
    }

    cout << "Error scalar potential: " << err << endl;
    REQUIRE( err < 1e-13 );

  }

  SECTION("advection_z") {
    gi.debug_adv_v = false; // disable in advection in v

    double t_final = 0.1;
    lr2<double> f_final = integration(t_final, 1e-3, gi, X, V); 

    mat K({gi.dxx_mult,gi.r});
    blas.matmul(f_final.X, f_final.S, K);
    mat f_full({gi.dxx_mult, gi.dzv_mult});
    blas.matmul_transb(K, f_final.V, f_full);
    
    double err = 0.0;
    for(Index jv=0;jv<gi.N_zv[1];jv++) {
      for(Index jz=0;jz<gi.N_zv[0];jz++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++) {
          for(Index ix=0;ix<gi.N_xx[0];ix++) {
            mfp<2> xy = gi.x({ix, iy});
            mfp<2> zv = gi.v({jz, jv});
            double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0]-t_final*zv[1])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1],2));
            double val = f_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            if(std::isnan(val)) 
              err = std::numeric_limits<double>::infinity();
            else
              err = max(err, abs(val - expected));
          }
        }
      }
    }
            
    cout << "Error advection z: " << err << endl;
    REQUIRE( err <= 3e-5 );
  }

  SECTION("advection_v") {
    gi.debug_adv_z = false;

    mat Kphi({gi.dxx_mult, gi.r});
    for(Index k=0;k<gi.r;k++) {
      for(Index j=0;j<gi.N_xx[1];j++) {
        for(Index i=0;i<gi.N_xx[0];i++) {
          Kphi(i+gi.N_xx[0]*j, k) = (k==0) ? sqrt(M_PI)*sqrt(2.0*M_PI) : 0.0;
        }
      }
    }

    mat Vphi({gi.N_zv[0], gi.r});
    for(Index k=0;k<gi.r;k++) {
      for(Index i=0;i<gi.N_zv[0];i++) {
        double z = gi.v(0, i);
        Vphi(i, k) = (k==0) ? sin(z)/sqrt(M_PI)/sqrt(2.0*M_PI) : 0.0;
      }
    }
    
    double t_final = 0.2;
    lr2<double> f_final = integration(t_final, 1e-2, gi, X, V, &Kphi, &Vphi); 

    mat K({gi.dxx_mult,gi.r});
    blas.matmul(f_final.X, f_final.S, K);
    mat f_full({gi.dxx_mult, gi.dzv_mult});
    blas.matmul_transb(K, f_final.V, f_full);
    
    double err = 0.0;
    for(Index jv=0;jv<gi.N_zv[1];jv++) {
      for(Index jz=0;jz<gi.N_zv[0];jz++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++) {
          for(Index ix=0;ix<gi.N_xx[0];ix++) {
            mfp<2> xy = gi.x({ix, iy});
            mfp<2> zv = gi.v({jz, jv});
            double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1]-t_final/gi.M_e*cos(zv[0]),2));
            double val = f_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            if(std::isnan(val)) 
              err = std::numeric_limits<double>::infinity();
            else
              err = max(err, abs(val - expected));
          }
        }
      }
    }
            
    cout << "Error advection v: " << err << endl;
    REQUIRE( err <= 5e-3 );
  }

}
*/

int main() {

  double rho_i = 1e-3;
  double kperp = 0.2/rho_i;
  double kx = sqrt(0.5)*kperp;
  double ky = sqrt(0.5)*kperp;
  double M_e = 1.0/1830.0;
  double beta = 0.00219;
  double C_P = 1.0/pow(rho_i,2);
  double C_A = beta/pow(rho_i,2);
  double kpar = 1.0;
  double Vmax = 6.0/sqrt(M_e);
  double alpha = 1e-3;

  Index r = 5;
  mind<2> N_xx = {100, 100};
  mind<2> N_zv = {100, 100};
  mfp<4> lim_xx = {0.0,2*M_PI/kx,0.0,2*M_PI/ky};
  mfp<4> lim_zv = {0.0,2*M_PI,-Vmax,Vmax};
  grid_info<2> gi(r, N_xx, N_zv, lim_xx, lim_zv, M_e, C_P, C_A); 


  vec xx1({gi.dxx_mult}), xx2({gi.dxx_mult});
  componentwise_vec_omp(gi.N_xx, [&xx1, &xx2, &gi, kx, ky](Index idx, array<Index,2> i) {
    mfp<2> x  = gi.x(i);
    xx1(idx) = 1.0;
    xx2(idx) = cos(kx*x[0])*cos(ky*x[1]);
  });

  vec vv1({gi.dzv_mult}), vv2({gi.dzv_mult});
  componentwise_vec_omp(gi.N_zv, [&vv1, &vv2, &gi, &alpha](Index idx, array<Index,2> i) {
    mfp<2> zv  = gi.v(i);
    vv1(idx) = sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
    vv2(idx) = alpha*cos(zv[0])*sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
  });
  vector<const double*> X, V;
  X.push_back(xx1.begin());
  X.push_back(xx2.begin());
  V.push_back(vv1.begin());
  V.push_back(vv2.begin());

  integration(20.0, 5e-5, gi, X, V);
}
