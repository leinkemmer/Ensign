#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/timer.hpp>
#include <generic/fft.hpp>
#include <generic/netcdf.hpp>

using namespace Ensign;
using namespace Ensign::Matrix;

template<size_t d> using mind = array<Index,d>;
template<size_t d> using mfp  = array<double,d>;

using vec  = multi_array<double,1>;
using cvec = multi_array<complex<double>,1>;
using mat  = multi_array<double,2>;

struct grid_info {
  Index r;
  Index n_x;
  mind<2> n_v;
  Index N_v;
  mfp<3> lim_a, lim_b;
  double h_x;
  mfp<2> h_v;

  double g, Bhat, q, m;

  grid_info(Index _r, Index _n_x, mind<2> _n_v, mfp<3> _lim_a, mfp<3> _lim_b, double _g, double _B, double _q, double _m, double Omega_i)
    : r(_r), n_x(_n_x), n_v(_n_v), lim_a(_lim_a), lim_b(_lim_b), g(_g), Bhat(_B), q(_q), m(_m) {

    Bhat *= q*Omega_i/m;

    // compute h_xx and h_zv
    h_x = (lim_b[0]-lim_a[0])/n_x;
    h_v[0] = (lim_b[1]-lim_a[1])/n_v[0];
    h_v[1] = (lim_b[2]-lim_a[2])/n_v[1];

    N_v = n_v[0]*n_v[1];
  }

  Index lin_idx_v(mind<2> i) const {
    return i[0] + n_v[0]*i[1];
  }

  double x(Index i) const {
    return lim_a[0] + i*h_x;
  }

  double v(size_t k, Index i) const {
    return lim_a[1+k] + i*h_v[k];
  }

  mfp<2> v(mind<2> i) const {
    mfp<2> out;
    for(size_t k=0;k<2;k++)
      out[k] = v(k, i[k]);
    return out;
  }
};

grid_info modify_r(grid_info gi, Index r) {
  gi.r = r;
  return gi;
}

double integrate_x(const vec& n, const grid_info& gi) {
  double s = 0.0;
  for(Index i=0;i<gi.n_x;i++)
    s += gi.h_x*n(i);
  return s;
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


struct rk4 {

  rk4(mind<2> n) : k1(n), k2(n), k3(n), k4(n), tmp(n) {
  }

  void step(double tau, mat& U, std::function<void(const mat&, mat&)> rhs) {
    gt::start("rk4");

    // k1
    rhs(U, k1);

    // k2
    #ifdef __OPENMP__
    #pragma omp parallel for simd
    #endif
    for(Index i=0;i<tmp.num_elements();i++) {
      tmp.data()[i] = U.data()[i] + 0.5*tau*k1.data()[i];
    }
    rhs(tmp, k2);

    // k3
    #ifdef __OPENMP__
    #pragma omp parallel for simd
    #endif
    for(Index i=0;i<tmp.num_elements();i++) {
      tmp.data()[i] = U.data()[i] + 0.5*tau*k2.data()[i];
    }
    rhs(tmp, k3);

    // k4
    #ifdef __OPENMP__
    #pragma omp parallel for simd
    #endif
    for(Index i=0;i<tmp.num_elements();i++) {
      tmp.data()[i] = U.data()[i] + tau*k3.data()[i];
    }
    rhs(tmp, k4);
    
    #ifdef __OPENMP__
    #pragma omp parallel for simd
    #endif
    for(Index i=0;i<U.num_elements();i++) {
      U.data()[i] += tau*(1.0/6.0*k1.data()[i] + 1.0/3.0*k2.data()[i] + 1.0/3.0*k3.data()[i] + 1.0/6.0*k4.data()[i]);
    }

    gt::stop("rk4");
  }

private:
  mat k1, k2, k3, k4, tmp;
};



struct vlasov {

  void step(double tau, const vec& E) {
    vec Ehat = E;
    Ehat *= gi.q/gi.m;

    // K step
    gt::start("coeff_C");
    compute_C1(f.V, f.V);
    compute_C2(f.V, f.V);
    compute_C3(f.V, f.V);
    compute_C4(f.V, f.V);
    compute_C5(f.V, f.V);
    gt::stop("coeff_C");

    blas.matmul(f.X,f.S,K);
    rk4_K.step(tau, K, [this, &Ehat](const mat& K, mat& Kout) {
      rhs_K(K, Kout, Ehat);
    });

    f.X = K;
    gt::start("orthogonalize");
    gs(f.X, f.S, gi.h_x);
    gt::stop("orthogonalize");

    // S step
    gt::start("coeff_D");
    compute_D1(f.X, f.X);
    compute_D2(f.X, f.X, Ehat);
    gt::stop("coeff_D");

    rk4_S.step(tau, f.S, [this](const mat& S, mat& Sout) {
      rhs_S(S, Sout);
      Sout *= -1.0; // projector splitting integrator runs S backward in time
    });


    // L step
    blas.matmul_transb(f.V,f.S,L);
    rk4_L.step(tau, L, [this](const mat& L, mat& Lout) {
      rhs_L(L, Lout);
    });
    
    f.V = L;
    gt::start("orthogonalize");
    gs(f.V, f.S, gi.h_v[0]*gi.h_v[1]);
    gt::stop("orthogonalize");
    transpose_inplace(f.S);
  }

  void rhs_K(const mat&K, mat& Kout, const vec& Ehat) {
    gt::start("rhs_K");

    mat Ktmp(K), Ktmp2(K);

    deriv_y(K, Ktmp);
    blas.matmul_transb(Ktmp, C1, Kout);
    Kout *= -1.0;

    blas.matmul_transb(K, C2, Ktmp);
    Ktmp *= gi.g;
    Kout -= Ktmp;

    ptw_mult_row(K, Ehat, Ktmp);
    blas.matmul_transb(Ktmp, C3, Ktmp2);
    Kout -= Ktmp2;
    
    Ktmp = K;
    Ktmp *= gi.Bhat;
    blas.matmul_transb(Ktmp, C4, Ktmp2);
    Kout -= Ktmp2;

    blas.matmul_transb(Ktmp, C5, Ktmp2);
    Kout += Ktmp2;

    gt::stop("rhs_K");
  }


  void rhs_L(const mat& L, mat& Lout) {
    gt::start("rhs_L");
/*
    // This is the reference implementation using vector operations
    // which however is slower than the implementation below.
    ptw_mult_row(L, v_y, Ltmp);
    blas.matmul_transb(Ltmp, D1, Lout);
    Lout *= -1.0;

    deriv_vx(L, Ltmp);
    ptw_mult_row(Ltmp, v_y, Ltmp2);
    Ltmp *= gi.g;
    Lout -= Ltmp;
    
    Ltmp2 *= gi.Bhat;
    Lout += Ltmp2;

    deriv_vy(L, Ltmp);
    blas.matmul_transb(Ltmp, D2, Ltmp2);
    Lout -= Ltmp2; 

    ptw_mult_row(Ltmp, v_x, Ltmp2);
    gt::start("L_vec_test");
    Ltmp2 *= gi.Bhat;
    Lout -= Ltmp2;
   */
    
    deriv_vy(L, Ltmp);
    blas.matmul_transb(Ltmp, D2, Ltmp2);
    
    blas.matmul_transb(L, D1, Ltmp);

    #pragma omp parallel for simd
    for(Index ir=0;ir<gi.r;ir++) {
      for(Index iv2=0;iv2<gi.n_v[1];iv2++) {
        for(Index iv1=0;iv1<gi.n_v[0];iv1++) {
          double L_dvx = (L(gi.lin_idx_v({(iv1+1)%gi.n_v[0], iv2}), ir) - L(gi.lin_idx_v({(iv1-1+gi.n_v[0])%gi.n_v[0], iv2}), ir))/(2.0*gi.h_v[0]);
          double L_dvy = (L(gi.lin_idx_v({iv1,(iv2+1)%gi.n_v[1]}), ir) - L(gi.lin_idx_v({iv1, (iv2-1+gi.n_v[1])%gi.n_v[1]}), ir))/(2.0*gi.h_v[1]);

          Lout(gi.lin_idx_v({iv1,iv2}), ir) =
            - gi.v(1, iv2)*Ltmp(gi.lin_idx_v({iv1,iv2}), ir)
            - gi.g*L_dvx
            - Ltmp2(gi.lin_idx_v({iv1,iv2}), ir)
            - gi.Bhat*gi.v(1, iv2)*L_dvx
            + gi.Bhat*gi.v(0, iv1)*L_dvy;
        }
      }
    }
    
    gt::stop("rhs_L");
  }

  void rhs_S(const mat& S, mat& Sout) {
    mat Stmp(S), Stmp2(S);

    blas.matmul(D1, S, Stmp);
    blas.matmul_transb(Stmp, C1, Sout);
    Sout *= -1.0;

    blas.matmul_transb(S, C2, Stmp);
    Stmp *= gi.g;
    Sout -= Stmp;

    blas.matmul(D2, S, Stmp);
    blas.matmul_transb(Stmp, C3, Stmp2);
    Sout -= Stmp2;

    blas.matmul_transb(S, C4, Stmp);
    Stmp *= gi.Bhat;
    Sout -= Stmp;

    blas.matmul_transb(S, C5, Stmp);
    Stmp *= gi.Bhat;
    Sout += Stmp;
  }


  void compute_C1(const mat& V1, const mat& V2) {
    ptw_mult_row(V2,v_y,Vtmp); // multiply by v_y
    coeff(V1, Vtmp, gi.h_v[0]*gi.h_v[1], C1, blas);
  }

  void compute_C2(const mat& V1, const mat& V2) {
    deriv_vx(V2, Vtmp);
    coeff(V1, Vtmp, gi.h_v[0]*gi.h_v[1], C2, blas);
  }
  
  void compute_C3(const mat& V1, const mat& V2) {
    deriv_vy(V2, Vtmp);
    coeff(V1, Vtmp, gi.h_v[0]*gi.h_v[1], C3, blas);
  }

  void compute_C4(const mat& V1, const mat& V2) {
    deriv_vx(V2, Vtmp);
    ptw_mult_row(Vtmp,v_y,Vtmp); // multiply by v_y
    coeff(V1, Vtmp, gi.h_v[0]*gi.h_v[1], C4, blas);
  }

  void compute_C5(const mat& V1, const mat& V2) {
    deriv_vy(V2, Vtmp);
    ptw_mult_row(Vtmp,v_x,Vtmp); // multiply by v_y
    coeff(V1, Vtmp, gi.h_v[0]*gi.h_v[1], C5, blas);
  }

  void compute_D1(const mat& X1, const mat& X2) {
    deriv_y(X2, Xtmp);
    coeff(X1, Xtmp, gi.h_x, D1, blas);
  }
  
  void compute_D2(const mat& X1, const mat& X2, const vec& Ehat) {
    ptw_mult_row(X2,Ehat,Xtmp); // multiply by Ehat
    coeff(X1, Xtmp, gi.h_x, D2, blas);
  }


  double compute_nh(vec& nout, vec& hxout, vec& hyout) {
    vec int_V({gi.r}), tmp({gi.n_x});
    mat K(f.X);
    mat Vtmp(f.V);
    blas.matmul(f.X,f.S,K);

    // n
    integrate(f.V,gi.h_v[0]*gi.h_v[1],int_V,blas);
    blas.matvec(K,int_V,nout);
    
    // hx
    ptw_mult_row(f.V, v_x, Vtmp);
    integrate(Vtmp,gi.h_v[0]*gi.h_v[1],int_V,blas);
    blas.matvec(K,int_V,hxout);
    
    // hy
    ptw_mult_row(f.V, v_y, Vtmp);
    integrate(Vtmp,gi.h_v[0]*gi.h_v[1],int_V,blas);
    blas.matvec(K,int_V,hyout);

    // kinetic energy
    ptw_mult_row(f.V, v_sq, Vtmp);
    integrate(Vtmp,gi.h_v[0]*gi.h_v[1],int_V,blas);
    blas.matvec(K,int_V,tmp);
    return gi.m*integrate_x(tmp, gi); 
  }


  void deriv_vx(const mat& in, mat& out) {
    gt::start("deriv_vx");
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index ir=0;ir<gi.r;ir++) {
      for(Index iv2=0;iv2<gi.n_v[1];iv2++)
        for(Index iv1=0;iv1<gi.n_v[0];iv1++) {
          out(gi.lin_idx_v({iv1,iv2}), ir) = (in(gi.lin_idx_v({(iv1+1)%gi.n_v[0], iv2}), ir) - in(gi.lin_idx_v({(iv1-1+gi.n_v[0])%gi.n_v[0], iv2}), ir))/(2.0*gi.h_v[0]);
      }
    }
    gt::stop("deriv_vx");
  }

  void deriv_vy(const mat& in, mat& out) {
    gt::start("deriv_vy");
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index ir=0;ir<gi.r;ir++) {
      for(Index iv2=0;iv2<gi.n_v[1];iv2++)
        for(Index iv1=0;iv1<gi.n_v[0];iv1++) {
          out(gi.lin_idx_v({iv1,iv2}), ir) = (in(gi.lin_idx_v({iv1,(iv2+1)%gi.n_v[1]}), ir) - in(gi.lin_idx_v({iv1, (iv2-1+gi.n_v[1])%gi.n_v[1]}), ir))/(2.0*gi.h_v[1]);
      }
    }
    gt::stop("deriv_vy");
  }


  void deriv_y(const mat& in, mat& out) {
      for(Index ir=0;ir<gi.r;ir++) {
        for(Index iy=0;iy<gi.n_x;iy++) {
          out(iy, ir) = (in((iy+1)%gi.n_x, ir) - in((iy-1+gi.n_x)%gi.n_x, ir))/(2.0*gi.h_x);
        }
      }
  }

  vlasov(grid_info _gi, vector<const double*> X0, vector<const double*> V0) : f(_gi.r, {_gi.n_x, _gi.N_v}), rk4_K({_gi.n_x,_gi.r}), rk4_S({_gi.r,_gi.r}), rk4_L({_gi.N_v,_gi.r}), gi(_gi), gs(&blas) {

    initialize(f, X0, V0, gi.h_x, gi.h_v[0]*gi.h_v[1], blas);

    K.resize({gi.n_x, gi.r});
    L.resize({gi.N_v, gi.r});
    
    Xtmp.resize({gi.n_x, gi.r});
    Vtmp.resize({gi.N_v, gi.r});
    Ltmp.resize({gi.N_v, gi.r});
    Ltmp2.resize({gi.N_v, gi.r});

    C1.resize({gi.r, gi.r});
    C2.resize({gi.r, gi.r});
    C3.resize({gi.r, gi.r});
    C4.resize({gi.r, gi.r});
    C5.resize({gi.r, gi.r});
    D1.resize({gi.r, gi.r});
    D2.resize({gi.r, gi.r});

    v_x.resize({gi.N_v});
    componentwise_vec_omp(gi.n_v, [this](Index idx, mind<2> i) {
      v_x(idx) = gi.v(0, i[0]);
    });
    
    v_y.resize({gi.N_v});
    componentwise_vec_omp(gi.n_v, [this](Index idx, mind<2> i) {
      v_y(idx) = gi.v(1, i[1]);
    });
    
    v_sq.resize({gi.N_v});
    componentwise_vec_omp(gi.n_v, [this](Index idx, mind<2> i) {
      v_sq(idx) = 0.5*(pow(gi.v(0, i[0]), 2) + pow(gi.v(1, i[1]), 2));
    });
  }

  lr2<double> f;
  mat K, L, Xtmp, Vtmp, Ltmp, Ltmp2;
  mat C1, C2, C3, C4, C5, D1, D2;
  vec v_x, v_y, v_sq;
  rk4 rk4_K, rk4_S, rk4_L;
  grid_info gi;
  blas_ops blas;
  orthogonalize gs;
};



Index freq(Index k, Index n) {
  if(k < n/2)
    return k;
  else if(k == n/2)
    return 0;
  else
    return k-n;
}

struct poisson {

  poisson(grid_info _gi_e, grid_info _gi_i) : gi_e(_gi_e), gi_i(_gi_i) {
    rho.resize({gi_e.n_x});
    E.resize({gi_e.n_x});
    Ehat.resize({gi_e.n_x/2+1});
  }

  void compute(const vec& n_e, const vec& n_i) {
    rho = n_e;
    rho *= gi_e.q;

    E = n_i; // E is used as a temporary here and is overwritten by the inverse FFT
    E *= gi_i.q;

    rho += E;

    if(fft == nullptr)
      fft = make_unique_ptr<fft1d<1>>(array<Index,1>({gi_e.n_x}), rho, Ehat);

    fft->forward(rho, Ehat);

    double nc = 1.0/double(gi_e.n_x);
    for(Index i=0;i<gi_e.n_x/2+1;i++) {
      Index mult_i = freq(i, gi_e.n_x);
      complex<double> lambda = complex<double>(0.0,2.0*M_PI/(gi_e.lim_b[0]-gi_e.lim_a[0])*i);
      Ehat(i) *= (i==0 && mult_i==0) ? 0.0 : lambda/pow(lambda,2)*nc;
    }

    fft->backward(Ehat, E);

    ee = 0.0;
    for(Index i=0;i<gi_e.n_x;i++)
      ee += 0.5*pow(E(i),2)*gi_e.h_x;
  }

  double compute_anomcoll(const vec& n_e, const vec& n_i, const vec& hy_e, const vec& hy_i) {
    double int_Ene = 0.0;
    double int_nu = 0.0;
    for(Index i=0;i<gi_e.n_x;i++) {
      int_Ene += E(i)*n_e(i);
      int_nu += n_e(i)*(hy_e(i)/n_e(i) - hy_i(i)/n_i(i));
    }
    return gi_e.q*int_Ene/(gi_e.m*int_nu);
  }

  vec rho, E; //, n_e, n_i, u_e, u_i, hx_e, hy_e, hx_i, hy_i;
  cvec Ehat;
  grid_info gi_e, gi_i;
  std::unique_ptr<fft1d<1>> fft;
  blas_ops blas;
  double ee;
};


void save_lr(string fn, const lr2<double>& lr_sol, const grid_info& gi) {
    nc_writer ncw(fn, {gi.n_x, gi.n_v[0], gi.n_v[1], gi.r}, {"y", "vx", "vy", "r"});
    ncw.add_var("r", {"r"});
    ncw.add_var("y", {"y"});
    ncw.add_var("vx", {"vx"});
    ncw.add_var("vy", {"vy"});
    ncw.add_var("X", {"r", "y"});
    ncw.add_var("S", {"r", "r"});
    ncw.add_var("V", {"r", "vy", "vx"});

    ncw.start_write_mode();

    vector<double> vec_r(gi.r);
    for(Index i=0;i<gi.r;i++)
      vec_r[i] = i;

    vector<double> vec_y(gi.n_x);
    for(Index i=0;i<gi.n_x;i++)
        vec_y[i] = gi.x(i);

    vector<double> vec_vx(gi.n_v[0]), vec_vy(gi.n_v[1]);
    for(Index i=0;i<gi.n_v[0];i++)
        vec_vx[i] = gi.v(0, i);
    for(Index i=0;i<gi.n_v[1];i++)
        vec_vy[i] = gi.v(1, i);

    ncw.write("r", vec_r.data());
    ncw.write("y", vec_y.data());
    ncw.write("vx", vec_vx.data());
    ncw.write("vy", vec_vy.data());

    ncw.write("X", lr_sol.X.data());
    ncw.write("S", lr_sol.S.data());
    ncw.write("V", lr_sol.V.data());
}