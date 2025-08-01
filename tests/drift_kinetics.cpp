#include "../examples/drift_kinetics_interpolatory.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>


blas_ops blas;

TEST_CASE( "Drift-kinetic", "[dk]" ) {

 SECTION("adv_phi") {
  Index n_adv=64, n_coeff=10;
  grid_info gi(3, 9, {1,1,n_adv}, {n_coeff,1}, {0,0,0,-1,0}, {1,1,2*M_PI,1,0}, 1.0);

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

  for(Index n=0;n<100;n++) {
    rk4(dt, gi, f, blas);
  
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
        err = max_err(err, abs(f_full(i,j)-cos(phi-gi.vp(j)*2*M_PI)));
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

}