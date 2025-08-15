#include "../examples/drift_kinetics_interpolatory.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>


blas_ops blas;


double rhs_advphi(const grid_info& gi, const multi_array<double,2>& X, const multi_array<double,2>& L, const multi_array<double,1> potential, Index i, Index j, Index k, Index l, Index m, Index ir) {
  //Index phi = gi.lin_idx_x({i,j,k});
  Index phi_p1 = gi.lin_idx_x({i,j,(k+1)%gi.n_x[2]});
  Index phi_m1 = gi.lin_idx_x({i,j,(k-1+gi.n_x[2])%gi.n_x[2]});
  Index idx_v = gi.lin_idx_v({l,m});
  return -gi.vpar(l)/gi.R0*(X(phi_p1,ir)-X(phi_m1,ir))/(2.0*gi.h_x[2])*L(idx_v,ir);
  //return (X(phi_p1,ir)-X(phi_m1,ir))/(2.0*gi.h_x[2])*L(idx_v,ir);
  //return -(X(phi,ir)-X(phi_m1,ir))/(gi.h_x[2])*L(idx_v,ir);
}


TEST_CASE( "Drift-kinetic", "[dk]" ) {

  SECTION("adv_phi") {
   Index n_adv=64, n_coeff=10;
   grid_info gi(3, 9, {1,1,n_adv}, {n_coeff,1}, {0,0,0,-1,0}, {1,1,2*M_PI,1,0}, 1.0, 1.0, 1.0, 1.0, 1.0);
 
   vec x1({gi.N_x});
   for(Index i=0;i<n_adv;i++) {
     double phi = gi.phi(i);
     x1(i) = cos(phi);
   }
   vector<const double*> X;
   X.push_back(x1.begin());
 
   vec v1({gi.N_v});
   for(Index i=0;i<n_coeff;i++) {
     v1(i) = 1.0;
   } 
   vector<const double*> V;
   V.push_back(v1.begin());
 
   lr2<double> f(gi.r, {gi.N_x, gi.N_v});
   initialize(f, X, V, 1.0, 1.0, blas);
 
   //ofstream fs("test1.data");
   //for(Index i=0;i<32;i++)
   //  fs << i << " " << f.X(i, 0) << " " << f.V(i,0) << endl;
   //fs.close();
 
   /*
   multi_array<double,2> f_full = f.full(blas);
   double err=0.0;
   ofstream fs("in.data");
   ofstream fs2("check1.data");
   for(Index j=0;j<n_coeff;j++) {
     for(Index i=0;i<n_adv;i++) {
       double phi = gi.phi(i);
       err = max_err(err, abs(f_full(i,j)-cos(phi)));
       fs << cos(phi-M_PI) << " ";
       if(j==0)
         fs2 << i << " " << cos(phi) <<  endl;
     }
     fs << endl;
   }
   cout << "err: " << err << endl;
   */
 
   Index num_steps = 100;
   double t_final = 2*M_PI;
   double dt = t_final/num_steps;
 
   vec potential; // not used
   for(Index n=0;n<100;n++) {
     rk4(dt, gi, f, rhs_advphi, potential, blas);
   
   /*
   {
     multi_array<double,2> f_full = f.full(blas);
     double err=0.0;
     ofstream fs("out.data");
     ofstream fs2("check2.data");
     for(Index j=0;j<n_coeff;j++) {
       for(Index i=0;i<n_adv;i++) {
         double phi = gi.phi(i);
         err = max_err(err, abs(f_full(i,j)-cos(phi-0*double(n+1)/64*2*M_PI)));
         fs << f_full(i,j) << " ";
         if(j==0)
           fs2 << i << " " << f_full(i,j) << endl;
       }
       fs << endl;
     }
     cout << "err: " << err << endl;
 
     if(err > 1e-12) {
       cout << "error after " << n << endl;
       exit(1);
     }
   }
     */
   }
 
   {
     multi_array<double,2> f_full = f.full(blas);
     double err=0.0;
     ofstream fs("out.data");
     ofstream fs2("check2.data");
     for(Index j=0;j<n_coeff;j++) {
       for(Index i=0;i<n_adv;i++) {
         double phi = gi.phi(i);
         err = max_err(err, abs(f_full(i,j)-cos(phi-gi.vpar(j)*2*M_PI)));
         fs << f_full(i,j) << " ";
         if(j==0)
           fs2 << i << " " << f_full(i,j) << endl;
       }
       fs << endl;
     }
     cout << "err: " << err << endl;
     REQUIRE(err < 2e-2);
   }
 
  }
 
 
 
  SECTION("rhs_driftkinetic") {
    grid_info gi(1, 5, {33,64,31}, {30,1}, {0.5,0,0,-6.0,0}, {5,2*M_PI,2*M_PI,6.0,1.0}, 10.0, 2.0, 0.3, 0.7, 1.8);

    multi_array<double,2> X({gi.N_x, gi.r});
    multi_array<double,1> potential({gi.N_x});
    for(Index k=0;k<gi.n_x[2];k++) {
      for(Index j=0;j<gi.n_x[1];j++) {
        for(Index i=0;i<gi.n_x[0];i++) {
          Index idx = gi.lin_idx_x({i,j,k});
          double rvar = gi.rvar(i);
          double theta = gi.theta(j);
          double phi = gi.phi(k);
          X(idx, 0) = 1.0 + exp(-pow(rvar-2.5,2)/0.5)*cos(theta + 2*phi);
          potential(idx) = cos(phi) + rvar + sin(2*theta);
        }
      }
    }

    multi_array<double,2> L({gi.N_v, gi.r});
    for(Index m=0;m<gi.n_v[1];m++) {
      for(Index l=0;l<gi.n_v[0];l++) {
        Index idx = gi.lin_idx_v({l,m});
        double vpar = gi.vpar(l);
        L(idx, 0) = exp(-0.5*pow(vpar,2));
      }
    }

    double err_cd2 = 0.0, err_cd4 = 0.0, err_upw3 = 0.0, max_val = 0.0;
    for(Index m=0;m<gi.n_v[1];m++) {
      for(Index l=0;l<gi.n_v[0];l++) {
        for(Index k=0;k<gi.n_x[2];k++) {
          for(Index j=0;j<gi.n_x[1];j++) {
            for(Index i=2;i<gi.n_x[0]-2;i++) {
              double rvar = gi.rvar(i);
              double theta = gi.theta(j);
              double phi = gi.phi(k);
              double vpar = gi.vpar(l);

              double expv = exp(-0.5*pow(vpar,2));
              double expr = exp(-2.0*pow(rvar-2.5,2));

              double adv_phi = 2.0*expr*expv*vpar*sin(theta + 2*phi)/(gi.R0 + rvar*cos(theta)); // ok
              double adv_vpar =expv*gi.q*vpar*(1.0+expr*cos(theta+2*phi))*sin(phi)/(gi.m*(gi.R0 + rvar*cos(theta))); //ok
              double adv_theta = -expr*expv*gi.q*sin(theta+2*phi)/(gi.B0*gi.m*rvar); // ok
              double adv_r = 8.0*expr*expv*gi.q*(rvar-2.5)*cos(2*theta)*cos(theta+2.0*phi)/(gi.B0*gi.m*rvar);

              double val_exact = adv_phi + adv_vpar + adv_theta + adv_r;
              double val_cd2 = rhs_driftkinetic_cd2(gi, X, L, potential, i, j, k, l, m, 0);
              err_cd2 = max_err(err_cd2, abs(val_cd2 - val_exact));
              double val_cd4 = rhs_driftkinetic_cd4(gi, X, L, potential, i, j, k, l, m, 0);
              err_cd4 = max_err(err_cd4, abs(val_cd4 - val_exact));
              double val_upw3 = rhs_driftkinetic_upwind3(gi, X, L, potential, i, j, k, l, m, 0);
              err_upw3 = max_err(err_upw3, abs(val_upw3 - val_exact));
              max_val = max(max_val, abs(val_exact));
            }
          }
        }
      }
    }

    cout << "err_cd2: " << err_cd2/max_val << " " << max_val << endl;
    cout << "err_cd4: " << err_cd4/max_val << " " << max_val << endl;
    cout << "err_upw3: " << err_upw3/max_val << " " << max_val << endl;
    REQUIRE(err_cd2/max_val < 4e-2);
    REQUIRE(err_cd4/max_val < 3e-3);
    REQUIRE(err_upw3/max_val < 1e-2);

    // compute the rhs of the quasi-neutrality equation 
    gi.n0 = [](double r) { return 1.0+r; };
    quasi_neutrality_solver qns(gi);
    qns.compute_rhs(X, L);

    double err_qn = 0.0, max_val_qn = 0.0;
    for(Index k=0;k<gi.n_x[2];k++) {
      for(Index j=0;j<gi.n_x[1];j++) {
        Index idx_rhs = j + gi.n_x[1]*k;
        double theta = gi.theta(j);
        double phi = gi.phi(k);
        for(Index i=0;i<gi.n_x[0];i++) {
          double rvar = gi.rvar(i);

          double expr = exp(-2.0*pow(rvar-2.5,2));
          double val_exact = sqrt(2.0*M_PI)/gi.n0(rvar)*(1.0 + expr*cos(theta+2*phi))*gi.B0/gi.m - 1.0;
          double val = qns.rhs(idx_rhs, i);
          err_qn = max_err(err_qn, abs(val_exact-val));
          max_val_qn = max(max_val_qn, val_exact);
        }
      }
    }

    cout << "err qn_rhs: " << err_qn/max_val_qn << " " << max_val_qn << endl;
    REQUIRE(err_qn/max_val_qn < 1e-7);
  }


  SECTION("quasi_neutrality") {
    grid_info gi(1, 5, {32,34,31}, {30,1}, {0.5,0,0,-6.0,0}, {5,2*M_PI,2*M_PI,6.0,1.0}, 10.0, 0.7, 1.3, 2.0, 0.5);

    gi.n0 = [](double r) { return pow(r,2); };
    gi.Te = [](double r) { return r; };

    quasi_neutrality_solver qns(gi);
  
    for(Index k=0;k<gi.n_x[2];k++) {
      for(Index j=0;j<gi.n_x[1];j++) {
        for(Index i=0;i<gi.n_x[0];i++) {
          Index idx_rhs = j + gi.n_x[1]*k;
          double rvar = gi.rvar(i);
          double theta = gi.theta(j);
          double phi = gi.phi(k);

          //qns.rhs(idx_rhs, i) = (rvar-4.0*pow(rvar,2)+(-20.0+pow(rvar,2)*(-3.0+gi.B0*gi.q*(-5.0+rvar)*(4.0+rvar)*gi.Omega))*cos(2*phi)*sin(theta))/(gi.B0*pow(rvar,2)*gi.Omega); // n0=Te=1
          qns.rhs(idx_rhs, i) = ((3.0-8.0*rvar)*rvar+(-20.0+rvar*(2.0-7.0*rvar+gi.B0*gi.q*(-5.0+rvar)*(4.0+rvar)*gi.Omega))*cos(2*phi)*sin(theta))/(gi.B0*pow(rvar,2)*gi.Omega);
        }
      }
    }

    multi_array<double,1> potential({gi.N_x});
    qns.solve(potential);

    double err=0.0, max_val=0.0;
    for(Index k=0;k<gi.n_x[2];k++) {
      for(Index j=0;j<gi.n_x[1];j++) {
        for(Index i=0;i<gi.n_x[0];i++) {
          double rvar = gi.rvar(i);
          double theta = gi.theta(j);
          double phi = gi.phi(k);

          double val_exact = (pow(rvar,2)-rvar-20.0)*(cos(2*phi)*sin(theta)+1.0);
          double val = potential(gi.lin_idx_x({i,j,k}));

          err = max_err(err, abs(val - val_exact));
          max_val = max(max_val, abs(val_exact));
        }
      }
    }

    cout << "err qn_solver: " << err/max_val << " " << max_val << endl;
    REQUIRE(err/max_val < 1e-7);
  }

}