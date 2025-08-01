#include <lr/lr.hpp>
#include <lr/interpolatory.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>

using namespace Ensign;
using namespace Ensign::Matrix;

template<size_t d> using mind = array<Index,d>;
template<size_t d> using mfp  = array<double,d>;

using vec  = multi_array<double,1>;
using cvec = multi_array<complex<double>,1>;
using mat  = multi_array<double,2>;

using indices = multi_array<Index,1>;

struct grid_info {
  Index r, r_over;
  mind<3> n_x;
  Index N_x;
  mind<2> n_v;
  Index N_v;
  mfp<5> lim_a, lim_b;
  double R0;
  array<double,3> h_x;
  array<double,2> h_v;

  grid_info(Index _r, Index _r_over, mind<3> _n_x, mind<2> _n_v, mfp<5> _lim_a, mfp<5> _lim_b, double _R0) 
  : r(_r), r_over(_r_over), n_x(_n_x), n_v(_n_v), lim_a(_lim_a), lim_b(_lim_b), R0(_R0) {
    for(Index i=0;i<3;i++) {
      h_x[i] = (lim_b[i]-lim_a[i])/n_x[i];
    }
    N_x = n_x[0]*n_x[1]*n_x[2];

    for(Index i=0;i<2;i++) {
      h_v[i] = (lim_b[3+i]-lim_a[3+i])/n_v[i];
    }
    N_v = n_v[0]*n_v[1];
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

  double vp(Index i) const {
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


double rhs(const grid_info& gi, const multi_array<double,2>& X, const multi_array<double,2>& L, Index i, Index j, Index k, Index l, Index m, Index ir) {
  Index phi = gi.lin_idx_x({i,j,k});
  Index phi_p1 = gi.lin_idx_x({i,j,(k+1)%gi.n_x[2]});
  Index phi_m1 = gi.lin_idx_x({i,j,(k-1+gi.n_x[2])%gi.n_x[2]});
  Index idx_v = gi.lin_idx_v({l,m});
  return -gi.vp(l)/gi.R0*(X(phi_p1,ir)-X(phi_m1,ir))/(2.0*gi.h_x[2])*L(idx_v,ir);
  //return (X(phi_p1,ir)-X(phi_m1,ir))/(2.0*gi.h_x[2])*L(idx_v,ir);
  //return -(X(phi,ir)-X(phi_m1,ir))/(gi.h_x[2])*L(idx_v,ir);
}

// TOOD: pointer of const vs const pointer
void compute_stage(double dt, const grid_info& gi, const multi_array<double,2>& X0, const multi_array<double,2>& L0, double fac0, const multi_array<double,2>& X, const multi_array<double,2>& L, const indices& I, const indices& J, multi_array<double,2>& f_I, multi_array<double,2>& f_J) {

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


void rk4(double dt, const grid_info& gi, lr2<double>& f, blas_ops& blas) {
  multi_array<Index,1> I({gi.r_over}), J({gi.r_over});
  deim_ext(f.X, gi.r_over, I);
  deim_ext(f.V, gi.r_over, J);
  
  multi_array<double,2> f_I({gi.r_over,gi.N_v}), f_J({gi.N_x, gi.r_over});
  multi_array<double,2> f_I_stage({gi.r_over,gi.N_v}), f_J_stage({gi.N_x, gi.r_over});
  multi_array<double,2> X(f.X.shape());
  multi_array<double,2> L0(f.V.shape()), L(f.V.shape());

  f_I.set_zero();
  f_J.set_zero();


  // // euler (for testing)
  // blas.matmul_transb(f.V, f.S, L0);
  // compute_stage(dt, gi, f.X, L0, 1.0, f.X, L0, I, J, f_I_stage, f_J_stage);
  // f_I.sadd(1.0, f_I_stage);
  // f_J.sadd(1.0, f_J_stage);

  // first stage of RK4
  blas.matmul_transb(f.V, f.S, L0);
  compute_stage(0.5*dt, gi, f.X, L0, 1.0, f.X, L0, I, J, f_I_stage, f_J_stage);
  f_I.sadd(1.0/3.0, f_I_stage);
  f_J.sadd(1.0/3.0, f_J_stage);

  // second stage of RK4
  colloquation_to_lr(f_I_stage, f_J_stage, I, X, L, blas);
  compute_stage(0.5*dt, gi, f.X, L0, 1.0, X, L, I, J, f_I_stage, f_J_stage);
  f_I.sadd(2.0/3.0, f_I_stage);
  f_J.sadd(2.0/3.0, f_J_stage);
  
  // third stage of RK4
  colloquation_to_lr(f_I_stage, f_J_stage, I, X, L, blas);
  compute_stage(dt, gi, f.X, L0, 1.0, X, L, I, J, f_I_stage, f_J_stage);
  f_I.sadd(1.0/3.0, f_I_stage);
  f_J.sadd(1.0/3.0, f_J_stage);

  // fourth stage of RK4
  colloquation_to_lr(f_I_stage, f_J_stage, I, X, L, blas);
  compute_stage(0.5*dt, gi, f.X, L0, -1.0, X, L, I, J, f_I_stage, f_J_stage);
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