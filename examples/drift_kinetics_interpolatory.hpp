#include <lr/lr.hpp>
#include <lr/interpolatory.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <generic/fft.hpp>
#include <generic/timer.hpp>
#include <generic/netcdf.hpp>

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
    //return i[1] + i[2]*n_x[1] + i[0]*n_x[1]*n_x[2];
  }
  
  Index lin_idx_v(mind<2> i) const {
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
    //return {idx/n_x[1]/n_x[2], idx%n_x[1], (idx/n_x[1])%n_x[2]};
  }
  
  array<Index,2> from_lin_idx_v(Index idx) const {
    return {idx % n_v[0], idx/n_v[0]};
  }

  Index shift_rvar(mind<3> idx, Index shift) const {
    if(idx[0]+shift >= 0 && idx[0]+shift < n_x[0]) {
      return lin_idx_x({idx[0]+shift,idx[1],idx[2]});
    } else {
      return lin_idx_x({(idx[0]+n_x[0]+shift)%n_x[0],idx[1],idx[2]});
    }
  }

  Index shift_theta(mind<3> idx, Index shift) const {
    if(idx[1]+shift >= 0 && idx[1]+shift < n_x[1]) {
      return lin_idx_x({idx[0],idx[1]+shift,idx[2]});
    } else {
      return lin_idx_x({idx[0],(idx[1]+n_x[1]+shift)%n_x[1],idx[2]});
    }
  }

  Index shift_phi(mind<3> idx, Index shift) const {
    if(idx[2]+shift >= 0 && idx[2]+shift < n_x[2]) {
      return lin_idx_x({idx[0], idx[1], idx[2]+shift});
    } else {
      return lin_idx_x({idx[0], idx[1], (idx[2]+n_x[2]+shift)%n_x[2]});
    }
  }

  Index shift_vpar(mind<2> idx, Index shift) const {
    if(idx[0]+shift >= 0 && idx[0]+shift < n_v[0]) {
      return lin_idx_v({idx[0]+shift,idx[1]});
    } else {
      return lin_idx_v({(idx[0]+n_v[0]+shift)%n_v[0], idx[1]});
    }
  }

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
      lu_solvers[j].A(0, 0) = -3.0/2.0/h;
      lu_solvers[j].A(0, 1) =  2.0/h;
      lu_solvers[j].A(0, 2) =  -1.0/2.0/h;
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
            double val = X(idx,ir)*L_int(ir)/n0_i;
            rhs(idx_rhs,i) += val;
          }
        }
      }
    }

  }

  void solve(vec& potential) {
    if(fft == nullptr)
      fft = make_unique_ptr<fft2d<2>>(array<Index,2>({gi.n_x[1],gi.n_x[2]}), rhs, rhs_hat);

    fft->forward(rhs, rhs_hat);
    
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


template<class Evaluator>
double cd2(Evaluator eval, double h) {
  return (eval(1)-eval(-1))/(2.0*h);
}

double rhs_driftkinetic_cd2(const grid_info& gi, const mat& X, const mat& L, const vec& potential, const vec& potential_rvar, const vec& potential_theta, Index i, Index j, Index k, Index l, Index m, Index ir) {
  double rvar = gi.rvar(i);
  double theta = gi.theta(j);
  double vpar = gi.vpar(l);

  double R = gi.R0 + rvar*cos(theta);

  mind<3> idx_x = {i,j,k};
  mind<2> idx_v = {l,m};
  Index lidx_x = gi.lin_idx_x(idx_x);
  Index lidx_v = gi.lin_idx_v(idx_v);

  if(i==0 || i==gi.n_x[0]-1) {
    return 0.0; 
  } else {
    double X_rvar = cd2([&](Index s) { return potential_theta(gi.shift_rvar(idx_x,s))*X(gi.shift_rvar(idx_x,s),ir); }, gi.h_x[0]);
    double X_theta = cd2([&](Index s) { return potential_rvar(gi.shift_theta(idx_x,s))*X(gi.shift_theta(idx_x,s),ir); }, gi.h_x[1]);
    double X_phi = cd2([&](Index s) { return X(gi.shift_phi(idx_x,s),ir); }, gi.h_x[2]);

    double potential_phi = cd2([&](Index s) { return potential(gi.shift_phi(idx_x,s)); }, gi.h_x[2]);

    double L_vpar = cd2([&](Index s) { return L(gi.shift_vpar(idx_v,s),ir); }, gi.h_v[0]);

    double adv_r = -gi.q/(gi.m*gi.B0*rvar)*X_rvar*L(lidx_v,ir);
    double adv_theta = +gi.q/(gi.m*gi.B0*rvar)*X_theta*L(lidx_v,ir);
    
    double adv_phi =  -vpar/R*X_phi*L(lidx_v,ir);
    double adv_vpar = gi.q/(gi.m*R)*potential_phi*X(lidx_x, ir)*L_vpar;
    return adv_r + adv_theta + adv_phi + adv_vpar;
  }

}


template<class Evaluator>
double cd4(Evaluator eval, double h) {
  return (1.0/12.0*eval(-2) - 2.0/3.0*eval(-1) + 2.0/3.0*eval(1) - 1.0/12.0*eval(2))/h;
}

double rhs_driftkinetic_cd4(const grid_info& gi, const mat& X, const mat& L, const vec& potential, const vec& potential_rvar, const vec& potential_theta, Index i, Index j, Index k, Index l, Index m, Index ir) {
  double rvar = gi.rvar(i);
  double theta = gi.theta(j);
  double vpar = gi.vpar(l);

  double R = gi.R0 + rvar*cos(theta);

  mind<3> idx_x = {i,j,k};
  mind<2> idx_v = {l,m};
  Index lidx_x = gi.lin_idx_x(idx_x);
  Index lidx_v = gi.lin_idx_v(idx_v);

  if(i==0 || i==1 || i==gi.n_x[0]-1 || i==gi.n_x[0]-2) {
    return 0.0; 
  } else {
    // this is not in conservative formulation
    double potential_rvar = cd4([&](Index i) { return potential(gi.shift_rvar(idx_x,i)); }, gi.h_x[0]);
    double potential_theta = cd4([&](Index i) { return potential(gi.shift_theta(idx_x,i)); }, gi.h_x[1]);
    double X_rvar = potential_theta*cd4([&](Index s) { return X(gi.shift_rvar(idx_x,s),ir); }, gi.h_x[0]);
    double X_theta = potential_rvar*cd4([&](Index s) { return X(gi.shift_theta(idx_x,s),ir); }, gi.h_x[1]);
    double X_phi = cd4([&](Index s) { return X(gi.shift_phi(idx_x,s),ir); }, gi.h_x[2]);

    double potential_phi = cd4([&](Index s) { return potential(gi.shift_phi(idx_x,s)); }, gi.h_x[2]);

    double L_vpar = cd4([&](Index s) { return L(gi.shift_vpar(idx_v,s),ir); }, gi.h_v[0]);

    double adv_r = -gi.q/(gi.m*gi.B0*rvar)*X_rvar*L(lidx_v,ir);
    double adv_theta = +gi.q/(gi.m*gi.B0*rvar)*X_theta*L(lidx_v,ir);
    
    double adv_phi =  -vpar/R*X_phi*L(lidx_v,ir);
    double adv_vpar = gi.q/(gi.m*R)*potential_phi*X(lidx_x, ir)*L_vpar;

    return adv_r + adv_theta + adv_phi + adv_vpar;
  }
}


template<class Evaluator>
double upwind3(Evaluator eval, double velocity, double h) {
  if(velocity > 0.0) {
    return (1.0/6.0*eval(-2) - eval(-1) + 0.5*eval(0) + 1.0/3.0*eval(1))/h;
  } else {
    return (-1.0/6.0*eval(2) + eval(1) - 0.5*eval(0) - 1.0/3.0*eval(-1))/h;
  }
}

double rhs_driftkinetic_upwind3(const grid_info& gi, const mat& X, const mat& L, const vec& potential, const vec& potential_rvar, const vec& potential_theta, Index i, Index j, Index k, Index l, Index m, Index ir) {
  double rvar = gi.rvar(i);
  double theta = gi.theta(j);
  double vpar = gi.vpar(l);

  double R = gi.R0 + rvar*cos(theta);

  mind<3> idx_x = {i,j,k};
  mind<2> idx_v = {l,m};
  Index lidx_x = gi.lin_idx_x(idx_x);
  Index lidx_v = gi.lin_idx_v(idx_v);

  if(i==0 || i==1 || i==gi.n_x[0]-1 || i==gi.n_x[0]-2) {
    return 0.0; 
  } else {
    // not in conservative form
    double potential_rvar = cd4([&](Index i) { return potential(gi.shift_rvar(idx_x,i)); }, gi.h_x[0]);
    double potential_theta = cd4([&](Index i) { return potential(gi.shift_theta(idx_x,i)); }, gi.h_x[1]);

    double adv_r_coeff = gi.q/(gi.m*gi.B0*rvar)*potential_theta;
    double X_rvar = potential_theta*upwind3([&](Index s) { return X(gi.shift_rvar(idx_x,s),ir); }, adv_r_coeff, gi.h_x[0]); 
    double adv_r = -gi.q/(gi.m*gi.B0*rvar)*X_rvar*L(lidx_v,ir);

    double adv_theta_coeff = -gi.q/(gi.m*gi.B0*rvar)*potential_rvar;
    double X_theta = potential_rvar*upwind3([&](Index s) { return X(gi.shift_theta(idx_x,s),ir); }, adv_theta_coeff, gi.h_x[1]);
    double adv_theta = gi.q/(gi.m*gi.B0*rvar)*X_theta*L(lidx_v,ir);

    double adv_phi_coeff = vpar/R;
    double X_phi = upwind3([&](Index s) { return X(gi.shift_phi(idx_x,s),ir); }, adv_phi_coeff, gi.h_x[2]);
    double adv_phi = -adv_phi_coeff*X_phi*L(lidx_v,ir);

    double potential_phi = cd4([&](Index s) { return potential(gi.shift_phi(idx_x,s)); }, gi.h_x[2]);
    double adv_vpar_coeff = -gi.q/(gi.m*R)*potential_phi;
    double L_vpar = upwind3([&](Index s) { return L(gi.shift_vpar(idx_v,s),ir); }, adv_vpar_coeff, gi.h_v[0]);
    double adv_vpar = -adv_vpar_coeff*X(lidx_x, ir)*L_vpar;

    return adv_r + adv_theta + adv_phi + adv_vpar;
  }
}


void save_lr(string fn, const lr2<double>& lr_sol, const vec& potential, const grid_info& gi) {
    nc_writer ncw(fn, {gi.n_x[0], gi.n_x[1], gi.n_x[2], gi.n_v[0], gi.n_v[1], gi.r}, {"rvar", "theta", "phi", "vpar", "mu", "r"});
    ncw.add_var("r", {"r"});
    ncw.add_var("rvar", {"rvar"});
    ncw.add_var("theta", {"theta"});
    ncw.add_var("phi", {"phi"});
    ncw.add_var("vpar", {"vpar"});
    ncw.add_var("mu", {"mu"});
    ncw.add_var("X", {"r", "phi", "theta", "rvar"});
    ncw.add_var("S", {"r", "r"});
    ncw.add_var("V", {"r", "mu", "vpar"});
    ncw.add_var("potential", {"phi", "theta", "rvar"});

    ncw.start_write_mode();

    vector<double> vec_r(gi.r);
    for(Index i=0;i<gi.r;i++)
      vec_r[i] = i;

    vector<double> vec_rvar(gi.n_x[0]);
    for(Index i=0;i<gi.n_x[0];i++)
        vec_rvar[i] = gi.rvar(i);

    vector<double> vec_theta(gi.n_x[1]);
    for(Index i=0;i<gi.n_x[1];i++)
        vec_theta[i] = gi.theta(i);
    
    vector<double> vec_phi(gi.n_x[2]);
    for(Index i=0;i<gi.n_x[2];i++)
        vec_phi[i] = gi.phi(i);

    vector<double> vec_vpar(gi.n_v[0]);
    for(Index i=0;i<gi.n_v[0];i++)
        vec_vpar[i] = gi.vpar(i);

    vector<double> vec_mu(gi.n_v[1]);
    for(Index i=0;i<gi.n_v[1];i++)
        vec_mu[i] = gi.mu(i);

    ncw.write("r", vec_r.data());
    ncw.write("rvar", vec_rvar.data());
    ncw.write("theta", vec_theta.data());
    ncw.write("phi", vec_phi.data());
    ncw.write("vpar", vec_vpar.data());
    ncw.write("mu", vec_mu.data());

    ncw.write("X", lr_sol.X.data());
    ncw.write("S", lr_sol.S.data());
    ncw.write("V", lr_sol.V.data());

    ncw.write("potential", potential.data());
}

void potential_deriv(const vec& potential, vec& potential_r, vec& potential_theta, const grid_info& gi) {
  #pragma omp parallel for
  for(Index k=0;k<gi.n_x[2];k++) {
    for(Index j=0;j<gi.n_x[1];j++) {
      for(Index i=0;i<gi.n_x[0];i++) {
        mind<3> idx_x = {i,j,k};
        Index lidx_x = gi.lin_idx_x(idx_x);
        if(gi.n_x[0]==1) { // necessary for some of the test cases
            potential_r(lidx_x) = 0.0;
        } else if(i==0) {
          potential_r(lidx_x) = (-3.0/2.0*potential(gi.lin_idx_x({0,j,k})) + 2.0*potential(gi.lin_idx_x({1,j,k})) - 0.5*potential(gi.lin_idx_x({2,j,k})))/gi.h_x[0];
        } else if(i==gi.n_x[0]-1) {
          potential_r(lidx_x) = (3.0/2.0*potential(gi.lin_idx_x({gi.n_x[0]-1,j,k})) - 2.0*potential(gi.lin_idx_x({gi.n_x[0]-2,j,k})) + 0.5*potential(gi.lin_idx_x({gi.n_x[0]-3,j,k})))/gi.h_x[0];
        } else {
          potential_r(lidx_x) = cd2([&](Index s) { return potential(gi.shift_rvar(idx_x,s)); }, gi.h_x[0]);
        }
        potential_theta(lidx_x) = cd2([&](Index s) { return potential(gi.shift_theta(idx_x,s)); }, gi.h_x[1]);
      }
    }
  }
}

template<class RHS>
void compute_stage(double dt, const grid_info& gi, const mat& X0, const mat& L0, double fac0, const mat& X, const mat& L, const indices& I, const indices& J, RHS rhs, const vec& potential, mat& f_I, mat& f_J) {
  gt::start("compute_E_from_pot");
  vec potential_r({gi.N_x}), potential_theta({gi.N_x});
  potential_deriv(potential, potential_r, potential_theta, gi);
  gt::stop("compute_E_from_pot");

  gt::start("rk4_rhs_J");
  // Here we colloquate at v, mu points and compute f_J
  #pragma omp parallel for collapse(2)
  for(Index ir=0;ir<gi.r_over;ir++) {
    for(Index k=0;k<gi.n_x[2];k++) {
      Index idx_J = J(ir);
      array<Index,2> iv = gi.from_lin_idx_v(idx_J);

      for(Index j=0;j<gi.n_x[1];j++) {
        for(Index i=0;i<gi.n_x[0];i++) {
          Index idx_x = gi.lin_idx_x({i,j,k});

          f_J(idx_x, ir) = 0.0;
          for(Index ir2=0;ir2<gi.r;ir2++) {
            f_J(idx_x, ir) += fac0*X0(idx_x,ir2)*L0(idx_J,ir2)
                              + dt*rhs(gi, X, L, potential, potential_r, potential_theta, i, j, k, iv[0], iv[1], ir2);
          }
        }
      }
    }
  }
  gt::stop("rk4_rhs_J");

  gt::start("rk4_rhs_I");
  // Here we colloquate at r, theta, phi points and compute f_I
  #pragma omp parallel for
  for(Index ir=0;ir<gi.r_over;ir++) {
    Index idx_I = I(ir);
    array<Index,3> ix = gi.from_lin_idx_x(idx_I);

    for(Index m=0;m<gi.n_v[1];m++) {
      for(Index l=0;l<gi.n_v[0];l++) {
        Index idx_v = gi.lin_idx_v({l,m});

        f_I(ir, idx_v) = 0.0;
        for(Index ir2=0;ir2<gi.r;ir2++) {
          f_I(ir, idx_v) += fac0*X0(idx_I,ir2)*L0(idx_v,ir2)
                            + dt*rhs(gi, X, L, potential, potential_r, potential_theta, ix[0], ix[1], ix[2], l, m, ir2);
        }
      }
    }
  }
  gt::stop("rk4_rhs_I");
}

template<class RHS>
double rk4(double dt, const grid_info& gi, lr2<double>& f, RHS rhs, const multi_array<double,1>& potential, blas_ops& blas) {
  indices I({gi.r_over}), J({gi.r_over});
  deim_ext(f.X, gi.r_over, I);
  deim_ext(f.V, gi.r_over, J);
  
  mat f_I({gi.r_over,gi.N_v}), f_J({gi.N_x, gi.r_over});
  mat f_I_stage({gi.r_over,gi.N_v}), f_J_stage({gi.N_x, gi.r_over});
  mat X(f.X.shape());
  mat L0(f.V.shape()), L(f.V.shape());

  f_I.set_zero();
  f_J.set_zero();

  // first stage of RK4
  blas.matmul_transb(f.V, f.S, L0);
  compute_stage(0.5*dt, gi, f.X, L0, 1.0, f.X, L0, I, J, rhs, potential, f_I_stage, f_J_stage);
  f_I.sadd(1.0/3.0, f_I_stage);
  f_J.sadd(1.0/3.0, f_J_stage);

  // second stage of RK4
  double cond1 = colloquation_to_lr(f_I_stage, f_J_stage, I, X, L, blas);
  compute_stage(0.5*dt, gi, f.X, L0, 1.0, X, L, I, J, rhs, potential, f_I_stage, f_J_stage);
  f_I.sadd(2.0/3.0, f_I_stage);
  f_J.sadd(2.0/3.0, f_J_stage);
  
  // third stage of RK4
  double cond2 = colloquation_to_lr(f_I_stage, f_J_stage, I, X, L, blas);
  compute_stage(dt, gi, f.X, L0, 1.0, X, L, I, J, rhs, potential, f_I_stage, f_J_stage);
  f_I.sadd(1.0/3.0, f_I_stage);
  f_J.sadd(1.0/3.0, f_J_stage);

  // fourth stage of RK4
  double cond3 = colloquation_to_lr(f_I_stage, f_J_stage, I, X, L, blas);
  compute_stage(0.5*dt, gi, f.X, L0, -1.0, X, L, I, J, rhs, potential, f_I_stage, f_J_stage);
  f_I.sadd(1.0/3.0, f_I_stage);
  f_J.sadd(1.0/3.0, f_J_stage);

  double cond4 = colloquation_to_lr(f_I, f_J, I, f, blas);

  return std::max({cond1, cond2, cond3, cond4});
}
