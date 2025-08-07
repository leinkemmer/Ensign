#include <lr/lr.hpp>
#include <lr/interpolatory.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <generic/fft.hpp>

using namespace Ensign;
using namespace Ensign::Matrix;

template<size_t d> using mind = array<Index,d>;
template<size_t d> using mfp  = array<double,d>;

using vec  = multi_array<double,1>;
using cvec = multi_array<complex<double>,1>;
using mat  = multi_array<double,2>;
using cmat = multi_array<complex<double>,2>;

using indices = multi_array<Index,1>;

struct grid_info {
  Index r, r_over;
  mind<3> n_x;
  Index N_x;
  mind<2> n_v;
  Index N_v;
  mfp<5> lim_a, lim_b;
  double R0, q, m, B0, Omega;
  array<double,3> h_x;
  array<double,2> h_v;

  std::function<double(double)> n0;
  std::function<double(double)> Te;
  std::function<double(double)> Ti;

  grid_info(Index _r, Index _r_over, mind<3> _n_x, mind<2> _n_v, mfp<5> _lim_a, mfp<5> _lim_b, double _R0, double _q, double _m, double _B0, double _Omega) 
  : r(_r), r_over(_r_over), n_x(_n_x), n_v(_n_v), lim_a(_lim_a), lim_b(_lim_b), R0(_R0), q(_q), m(_m), B0(_B0), Omega(_Omega) {
    h_x[0] = (lim_b[0]-lim_a[0])/(n_x[0]-1); // both boundary points are included in r
    for(Index i=1;i<3;i++) {
      h_x[i] = (lim_b[i]-lim_a[i])/n_x[i];
    }
    N_x = n_x[0]*n_x[1]*n_x[2];

    for(Index i=0;i<2;i++) {
      h_v[i] = (lim_b[3+i]-lim_a[3+i])/n_v[i];
    }
    N_v = n_v[0]*n_v[1];

    // default value for n0, Te, Ti
    n0 = [](double) { return 1.0; };
    Te = [](double) { return 1.0; };
    Ti = [](double) { return 1.0; };
  }

  double feq(double rvar, double vpar) const {
    return n0(rvar)*exp(-pow(vpar,2)/(2.0*Ti(rvar)));
  }

  Index lin_idx_x(mind<3> i) const {
    return i[0] + n_x[0]*i[1] + n_x[0]*n_x[1]*i[2];
  }
  
  Index lin_idx_v(mind<3> i) const {
    return i[0] + n_v[0]*i[1];
  }

  double rvar(Index i) const {
    return lim_a[0] + i*h_x[0];
  }
  
  double theta(Index i) const {
    return lim_a[1] + i*h_x[1];
  }
  
  double phi(Index i) const {
    return lim_a[2] + i*h_x[2];
  }

  double vpar(Index i) const {
    return lim_a[3] + i*h_v[0];
  }

  double mu(Index i) const {
    return lim_a[4] + i*h_v[1];
  }

  array<Index,3> from_lin_idx_x(Index idx) const {
    return {idx % n_x[0], (idx/n_x[0]) % n_x[1], (idx/n_x[0]/n_x[1])};
  }
  
  array<Index,2> from_lin_idx_v(Index idx) const {
    return {idx % n_v[0], idx/n_v[0]};
  }

  /*
  template<size_t k>
  Index x_from_idx(Index idx) {
    if constexpr (k==0) {
        return lim_a[0] + (idx % n_x[0])*h_x[0]; 
    } else if constexpr (k==1) {
        return lim_a[1] + ((idx/n_x[0]) % n_x[1])*h_x[1]; 
    } else {
        return lim_a[2] + (idx/n_x[0]/n_x[1])*h_x[2]; 
    }
  }
  
  template<size_t k>
  Index v_from_idx(Index idx) {
    if constexpr (k==0) {
        return lim_a[3] + (idx % n_v[0])*h_v[0]; 
    } else {
        return lim_a[4] + ((idx/n_v[0]) % n_v[1])*h_v[1]; 
    }
  }
    */
};

struct quasi_neutrality_solver {

  quasi_neutrality_solver(grid_info _gi) 
    : rhs({_gi.n_x[1]*_gi.n_x[2],_gi.n_x[0]}), gi(_gi), lu_solvers(_gi.n_x[1]+1, lu_solver<complex<double>>(_gi.n_x[0])) {

    N_hat = gi.n_x[2]*(gi.n_x[1]/2 + 1);
    rhs_hat.resize({N_hat,gi.n_x[0]});

    // setup the matrices for the 1d linear solve
    double h = gi.h_x[0];
    for(Index j=0;j<gi.n_x[1]+1;j++) {
      lu_solvers[j].A.set_zero();
      for(Index i=1;i<gi.n_x[0]-1;i++) {
        double fac = -1.0/(gi.B0*gi.Omega);
        double rvar = gi.rvar(i);
        lu_solvers[j].A(i,i)   = fac*(-2.0/pow(h,2) - (j!=gi.n_x[1])*pow(j,2)/pow(rvar,2)) + (j!=gi.n_x[1])*gi.q/gi.Te(rvar);
        lu_solvers[j].A(i,i+1) = fac*(1.0/pow(h,2)  + (1.0/rvar + (gi.n0(rvar+1e-7)-gi.n0(rvar))/1e-7/gi.n0(rvar))/(2.0*h));
        lu_solvers[j].A(i,i-1) = fac*(1.0/pow(h,2)  - (1.0/rvar + (gi.n0(rvar+1e-7)-gi.n0(rvar))/1e-7/gi.n0(rvar))/(2.0*h));
      }
      // homogeneous Neumann boundary conditions on the left
      lu_solvers[j].A(0, 0) = -3.0/2.0;
      lu_solvers[j].A(0, 1) =  2.0;
      lu_solvers[j].A(0, 2) =  -1.0/2.0;
      // homogeneous Dirichlet condition on the right, but the boundary value is stored so we just leave it alone
      lu_solvers[j].A(gi.n_x[0]-1,gi.n_x[0]-1) = 1.0;
    }

    // precompute the LU decomposition
    for(Index j=0;j<gi.n_x[1]+1;j++) {
      lu_solvers[j].lu();
    }

  }

  void compute_rhs(const mat& X, const mat& L) {
    Index r = L.shape()[1];

    // integrate L
    vec L_int({r});
    for(Index ir=0;ir<r;ir++) {
      L_int(ir) = 0.0;
      for(Index j=0;j<gi.N_v;j++) {
          L_int(ir) += L(j, ir)*gi.h_v[0]*gi.h_v[1];
      }
      L_int(ir) *= gi.B0/gi.m;
    }

    // compute rhs
    for(Index k=0;k<gi.n_x[2];k++) {
      for(Index j=0;j<gi.n_x[1];j++) {
        Index idx_rhs = j + gi.n_x[1]*k;
        for(Index i=0;i<gi.n_x[0];i++) {
          Index idx = gi.lin_idx_x({i,j,k});
          double n0_i = gi.n0(gi.rvar(i));
          rhs(idx_rhs,i) = -1.0;
          for(Index ir=0;ir<r;ir++) {
            rhs(idx_rhs,i) += X(idx,ir)*L_int(ir)/n0_i;
          }
        }
      }
    }

  }

  void solve(vec& potential) {
    if(fft == nullptr)
      fft = make_unique_ptr<fft2d<2>>(array<Index,2>({gi.n_x[1],gi.n_x[2]}), rhs, rhs_hat);

    fft->forward(rhs, rhs_hat);
/*
    // solve the linear systems
    cvec b({gi.n_x[0]});
    for(Index i=0;i<gi.n_x[0];i++) {
      b(i) = rhs_hat(0,i)/double(gi.n_x[1]*gi.n_x[2]);
    }
    b(0) = 0.0;
    b(gi.n_x[0]-1) = 0.0;
    cout << b << endl;
    lu_solvers[gi.n_x[1]].solve(b);
    cout << b << endl;
    //exit(1);
    for(Index i=0;i<gi.n_x[0];i++) {
      rhs_hat(0,i) = b(i);
    }
*/
    cvec b({gi.n_x[0]});
    for(Index k=0;k<gi.n_x[2];k++) {
      for(Index j=0;j<gi.n_x[1]/2+1;j++) {
        Index idx_solver = j;
        if(k==0 && j==0) {
          idx_solver = gi.n_x[1];
        }

        // add the \hat{phi}_{00} to the rhs
        Index idx_rhs = j + (gi.n_x[1]/2+1)*k;
        for(Index i=0;i<gi.n_x[0];i++) {
          // TODO: something is not correct here
          b(i) = rhs_hat(idx_rhs,i)/double(gi.n_x[1]*gi.n_x[2]) + /*TODO*/0.0*gi.q/gi.Te(gi.rvar(i))*rhs_hat(0,i);
        }
        b(0) = 0.0;
        b(gi.n_x[0]-1) = 0.0;
        //cout << b << endl;
        lu_solvers[idx_solver].solve(b);
        //cout << b << endl;
        //exit(1);
        for(Index i=0;i<gi.n_x[0];i++) {
          rhs_hat(idx_rhs,i) = b(i);
        }
      }
    }

    fft->backward(rhs_hat, rhs);

    // copy to the vec format
    for(Index j=0;j<gi.n_x[1];j++) {
      for(Index k=0;k<gi.n_x[2];k++) {
        for(Index i=0;i<gi.n_x[0];i++) {
          Index idx = gi.lin_idx_x({i,j,k});
          Index idx_rhs = j + gi.n_x[1]*k;
          potential(idx) = rhs(idx_rhs,i);
        }
      }
    }

  }

  mat rhs; // this is in the format required by the FFT
private:
  grid_info gi;
  Index N_hat;

  std::unique_ptr<fft2d<2>> fft;

  cmat rhs_hat;
  vector<lu_solver<complex<double>>> lu_solvers;
};


double rhs_driftkinetic(const grid_info& gi, const mat& X, const mat& L, const vec& potential, Index i, Index j, Index k, Index l, Index m, Index ir) {
  double rvar = gi.rvar(i);
  double theta = gi.theta(j);
  double vpar = gi.vpar(l);

  double R = gi.R0 + rvar*cos(theta);

  Index idx_x = gi.lin_idx_x({i,j,k});
  Index idx_v = gi.lin_idx_v({l,m});
  
  Index rvar_p1 = gi.lin_idx_x({(i+1)%gi.n_x[0],j,k});
  Index rvar_m1 = gi.lin_idx_x({(i-1+gi.n_x[0])%gi.n_x[0],j,k});
  
  Index theta_p1 = gi.lin_idx_x({i,(j+1)%gi.n_x[1],k});
  Index theta_m1 = gi.lin_idx_x({i,(j-1+gi.n_x[1])%gi.n_x[1],k});
  
  Index phi_p1 = gi.lin_idx_x({i,j,(k+1)%gi.n_x[2]});
  Index phi_m1 = gi.lin_idx_x({i,j,(k-1+gi.n_x[2])%gi.n_x[2]});

  Index vpar_p1 = gi.lin_idx_v({(l+1)%gi.n_v[0], m});
  Index vpar_m1 = gi.lin_idx_v({(l-1+gi.n_v[0])%gi.n_v[0], m});

  //double adv_r, adv_theta;
  //if(i == 0) {
  //  adv_r = -gi.q/(gi.m*gi.B0*rvar)*(potential(theta_p1)-potential(theta_m1))/(2.0*gi.h_x[1])*(X(rvar_p1,ir)-X(idx_x,ir))/gi.h_x[0]*L(idx_v,ir);
  //  adv_theta = gi.q/(gi.m*gi.B0*rvar)*(potential(rvar_p1)-potential(idx_x))/gi.h_x[0]*(X(theta_p1,ir)-X(theta_m1,ir))/(2.0*gi.h_x[1])*L(idx_v,ir);
  //} else if(i == gi.n_x[0]-1) {
  //  adv_r = -gi.q/(gi.m*gi.B0*rvar)*(potential(theta_p1)-potential(theta_m1))/(2.0*gi.h_x[1])*(X(idx_v,ir)-X(rvar_m1,ir))/(gi.h_x[0])*L(idx_v,ir);
  //  adv_theta = gi.q/(gi.m*gi.B0*rvar)*(potential(idx_x)-potential(rvar_m1))/(gi.h_x[0])*(X(theta_p1,ir)-X(theta_m1,ir))/(2.0*gi.h_x[1])*L(idx_v,ir);
  //} else {
  //  adv_r = -gi.q/(gi.m*gi.B0*rvar)*(potential(theta_p1)-potential(theta_m1))/(2.0*gi.h_x[1])*(X(rvar_p1,ir)-X(rvar_m1,ir))/(2.0*gi.h_x[0])*L(idx_v,ir);
  //  adv_theta = gi.q/(gi.m*gi.B0*rvar)*(potential(rvar_p1)-potential(rvar_m1))/(2.0*gi.h_x[0])*(X(theta_p1,ir)-X(theta_m1,ir))/(2.0*gi.h_x[1])*L(idx_v,ir);
  //}
  if(i==0 || i==gi.n_x[0]-1) {
    return 0.0; 
  } else {
    double adv_r = -gi.q/(gi.m*gi.B0*rvar)*(potential(theta_p1)-potential(theta_m1))/(2.0*gi.h_x[1])*(X(rvar_p1,ir)-X(rvar_m1,ir))/(2.0*gi.h_x[0])*L(idx_v,ir);
    double adv_theta = gi.q/(gi.m*gi.B0*rvar)*(potential(rvar_p1)-potential(rvar_m1))/(2.0*gi.h_x[0])*(X(theta_p1,ir)-X(theta_m1,ir))/(2.0*gi.h_x[1])*L(idx_v,ir);
    double adv_phi =  -vpar/R*(X(phi_p1,ir)-X(phi_m1,ir))/(2.0*gi.h_x[2])*L(idx_v,ir);
    double adv_vpar = gi.q/(gi.m*R)*(potential(phi_p1)-potential(phi_m1))/(2.0*gi.h_x[2])*X(idx_x, ir)*(L(vpar_p1,ir) - L(vpar_m1,ir))/(2.0*gi.h_v[0]);
  return adv_r + adv_theta + adv_phi + adv_vpar;
  }

}

// TOOD: pointer of const vs const pointer
template<class RHS>
void compute_stage(double dt, const grid_info& gi, const mat& X0, const mat& L0, double fac0, const mat& X, const mat& L, const indices& I, const indices& J, RHS rhs, mat& f_I, mat& f_J) {

  // Here we colloquate at v, mu points and compute f_J
  for(Index ir=0;ir<gi.r_over;ir++) {
    Index idx_J = J(ir);
    array<Index,2> iv = gi.from_lin_idx_v(idx_J);

    for(Index k=0;k<gi.n_x[2];k++) {
      for(Index j=0;j<gi.n_x[1];j++) {
        for(Index i=0;i<gi.n_x[0];i++) {
          Index idx_x = gi.lin_idx_x({i,j,k});

          f_J(idx_x, ir) = 0.0;
          for(Index ir2=0;ir2<gi.r;ir2++) {
            f_J(idx_x, ir) += fac0*X0(idx_x,ir2)*L0(idx_J,ir2)
                              + dt*rhs(gi, X, L, i, j, k, iv[0], iv[1], ir2);
          }
        }
      }
    }
  }

  // Here we colloquate at r, theta, phi points and compute f_I
  for(Index ir=0;ir<gi.r_over;ir++) {
    Index idx_I = I(ir);
    array<Index,3> ix = gi.from_lin_idx_x(idx_I);

    for(Index m=0;m<gi.n_v[1];m++) {
      for(Index l=0;l<gi.n_v[0];l++) {
        Index idx_v = gi.lin_idx_v({l,m});

        f_I(ir, idx_v) = 0.0;
        for(Index ir2=0;ir2<gi.r;ir2++) {
          f_I(ir, idx_v) += fac0*X0(idx_I,ir2)*L0(idx_v,ir2)
                            + dt*rhs(gi, X, L, ix[0], ix[1], ix[2], l, m, ir2);
        }
      }
    }
  }
}

template<class RHS>
void rk4(double dt, const grid_info& gi, lr2<double>& f, RHS rhs, blas_ops& blas) {
  indices I({gi.r_over}), J({gi.r_over});
  deim_ext(f.X, gi.r_over, I);
  deim_ext(f.V, gi.r_over, J);
  
  mat f_I({gi.r_over,gi.N_v}), f_J({gi.N_x, gi.r_over});
  mat f_I_stage({gi.r_over,gi.N_v}), f_J_stage({gi.N_x, gi.r_over});
  mat X(f.X.shape());
  mat L0(f.V.shape()), L(f.V.shape());

  f_I.set_zero();
  f_J.set_zero();


  // // euler (for testing)
  // blas.matmul_transb(f.V, f.S, L0);
  // compute_stage(dt, gi, f.X, L0, 1.0, f.X, L0, I, J, f_I_stage, f_J_stage);
  // f_I.sadd(1.0, f_I_stage);
  // f_J.sadd(1.0, f_J_stage);

  // first stage of RK4
  blas.matmul_transb(f.V, f.S, L0);
  compute_stage(0.5*dt, gi, f.X, L0, 1.0, f.X, L0, I, J, rhs, f_I_stage, f_J_stage);
  f_I.sadd(1.0/3.0, f_I_stage);
  f_J.sadd(1.0/3.0, f_J_stage);

  // second stage of RK4
  colloquation_to_lr(f_I_stage, f_J_stage, I, X, L, blas);
  compute_stage(0.5*dt, gi, f.X, L0, 1.0, X, L, I, J, rhs, f_I_stage, f_J_stage);
  f_I.sadd(2.0/3.0, f_I_stage);
  f_J.sadd(2.0/3.0, f_J_stage);
  
  // third stage of RK4
  colloquation_to_lr(f_I_stage, f_J_stage, I, X, L, blas);
  compute_stage(dt, gi, f.X, L0, 1.0, X, L, I, J, rhs, f_I_stage, f_J_stage);
  f_I.sadd(1.0/3.0, f_I_stage);
  f_J.sadd(1.0/3.0, f_J_stage);

  // fourth stage of RK4
  colloquation_to_lr(f_I_stage, f_J_stage, I, X, L, blas);
  compute_stage(0.5*dt, gi, f.X, L0, -1.0, X, L, I, J, rhs, f_I_stage, f_J_stage);
  f_I.sadd(1.0/3.0, f_I_stage);
  f_J.sadd(1.0/3.0, f_J_stage);

  /*
  ofstream fs("f_I.data");
    for(Index j=0;j<gi.N_v;j++) {
      fs << j << " ";
      for(Index i=0;i<gi.r_over;i++) {
        fs << f_I(i, j) << " ";
    }
    fs << endl;
  }
  
  {
  ofstream fs("f_J.data");
    for(Index j=0;j<gi.N_x;j++) {
      fs << j << " ";
      for(Index i=0;i<gi.r_over;i++) {
        fs << f_J(j, i) << " ";
    }
    fs << endl;
  }
  }
  */

  /*
  // compute the orthonormalized low-rank decomposition
  colloquation_to_lr(f_I, f_J, I, J, f.X, L, blas);

  {
  ofstream fs("f_X.data");
    for(Index j=0;j<gi.N_x;j++) {
      fs << j << " ";
      for(Index i=0;i<gi.r;i++) {
        fs << f.X(j,i) << " ";
    }
    fs << endl;
  }
  }

  {
    //multi_array<double,2> L({gi.N_v, gi.r});
    //blas.matmul_transb(f.V, f.S, L);
    ofstream fs("f_V.data");
    for(Index j=0;j<gi.N_v;j++) {
      fs << j << " ";
      for(Index i=0;i<gi.r;i++) {
        fs << L(j,i) << " ";
    }
    fs << endl;
  }
  }

  //static int i=0;
  //i++;
  //if(i >= 27) {
  //  exit(1);
  //}

  cout << "I: " << I << endl;
  cout << "J: " << J << endl;
  */
  colloquation_to_lr(f_I, f_J, I, f, blas);

}




/*
void phi_adv(double dt, const grid_info& gi, const lr2<double>& f, const mind<1>& I, const mind<1>& J, multi_array<double,2>& YI, multi_array<double,2>& YJ) {

  // TODO
  blas.matmul(f.X, f.S, K);
  V_J = extract_indices(V, J);
  blas.matmul(K, V_J, f_J);

  Index nad = gi.n_x[2];
  for(Index m=0;m<I.size();m++) {
    double v_par = gi.v_from_idx<0>(I[m]);
    double adv = -dt*v_par/gi.h_v[0];
    Index adv_n = Index(floor(adv));
    double alpha = adv - adv_n;
    for(Index i=0;i<gi.n_x[0];i++) {
      for(Index j=0;j<gi.n_x[1];j++) {
       // setup the interpolation
       for(Index k=0;k<gi.n_x[2];k++) {
        Index idx_x = gi.lin_idx_x({i,j,k});
        Index idx_x_0 = gi.lin_idx_x({i,j,(k+adv_n+nad)%nad});
        Index idx_x_1 = gi.lin_idx_x({i,j,(k+adv_n+1+nad)%nad});
        YJ(idx_x, I[m]) = alpha*f_J(idx_x_0,m) + (1.0-alpha)*f_J(idx_x_1,m);
       }
      }
    }
  }

  // TODO: the one for YI is just the opposite
  K_I = extract_indices(K, I);
  blas.matmul(K_I, V, f_I);

}
*/