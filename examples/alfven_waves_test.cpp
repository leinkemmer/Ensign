#include "alfven_waves.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>



TEST_CASE( "Alfven waves", "[alfven_waves]" ) {

  blas_ops blas;
    
  double C_P = 3.0;
  double C_A = 0.5;
  double M_e = 0.1;
  Index r=5;
  mind<2> N_xx = {32, 32};
  mind<2> N_zv = {64, 64};
  mfp<4> lim_xx = {0.0,2*M_PI,0.0,2*M_PI};
  mfp<4> lim_zv = {0.0,2*M_PI,-6.0/sqrt(M_e), 6.0/sqrt(M_e)};
  grid_info<2> gi(r, N_xx, N_zv, lim_xx, lim_zv, M_e, C_P, C_A, discretization::lw); 

  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(gi.h_xx[0]*gi.h_xx[1], gi.dxx_mult);
  std::function<double(double*,double*)> ip_zv = inner_product_from_const_weight(gi.h_zv[0]*gi.h_zv[1], gi.dzv_mult);
  std::function<double(double*,double*)> ip_z = inner_product_from_const_weight(gi.h_zv[0], gi.N_zv[0]);

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

  /*
  *  initialization
  */
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


  /*
  *  Computation of the scalar potential
  */ 
  SECTION("scalar_potential"){

    // scalar potential
    mat K({gi.dxx_mult,gi.r});
    blas.matmul(f.X, f.S, K);
    
    mat Kphi({gi.dxx_mult,gi.r});
    mat Vphi({gi.N_zv[0],gi.r});
    scalar_potential compute_phi(gi, blas);
    
    mat Kmrho({gi.dxx_mult,gi.r});
    compute_phi(K, f.V, Kphi, Vphi, &Kmrho);

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


  /*
  *  Computation of the vector potential
  */ 
  SECTION("vector_potential") {

    // vector potential
    mat KA({gi.dxx_mult, gi.r});
    mat VA({gi.N_zv[0], gi.r});
    vector_potential compute_A(gi, blas);
    compute_A(f, KA, VA);

    mat A({gi.dxx_mult, gi.N_zv[0]});
    blas.matmul_transb(KA, VA, A);

    double err = 0.0;
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++) {
        for(Index ix=0;ix<gi.N_xx[0];ix++) {
          mfp<2> xy = gi.x({ix, iy});
          double z = gi.v(0, jz);
          double expected = -gi.C_A*0.0;
          err = max(err, abs(A(ix+gi.N_xx[0]*iy,jz) - expected));
        }
      }
    }
    cout << "Error scalar potential: " << err << endl;
    REQUIRE( err < 1e-13 );
  }


  /*
  *  Iterative solver for the vector potential
  */

  SECTION("dtA_iterative_solver") {
    // initialize rho
    vec xx1({gi.dxx_mult}), xx2({gi.dxx_mult});
    componentwise_vec_omp(gi.N_xx, [&xx1, &xx2, &gi](Index idx, array<Index,2> i) {
      mfp<2> x  = gi.x(i);
      xx1(idx) = 1.0;
      xx2(idx) = cos(x[0])*cos(x[1]);
    });

    vec vv1({gi.N_zv[0]}), vv2({gi.N_zv[0]});
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      double z = gi.v(0, jz);
      vv1(jz) = 1.0;
      vv2(jz) = 0.5*cos(z);
    }

    lr2<double> rho(gi.r, {gi.dxx_mult, gi.N_zv[0]});
    vector<const double*> X = {xx1.data(), xx2.data()};
    vector<const double*> V = {vv1.data(), vv2.data()};
    initialize(rho, X, V, ip_xx, ip_z, blas);

    // set E to zero
    lr2<double> E(gi.r, {gi.dxx_mult, gi.N_zv[0]});
    memset(E.S.data(), 0, E.rank()*E.rank()*sizeof(double));

    // check the rhs
    dtA_iterative_solver dtA_it(gi, blas);
    dtA_it.compute_rhs(f, E, rho);
    mat dtA_full = dtA_it.rhs.full(blas);

    double err = 0.0;
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++) {
        for(Index ix=0;ix<gi.N_xx[0];ix++) {
          mfp<2> xy = gi.x({ix, iy});
          double z = gi.v(0, jz);
          double expected = -gi.C_A*cos(xy[0])*cos(xy[1])*sin(z)/(4.0*gi.M_e);
          err = max(err, abs(dtA_full(ix+gi.N_xx[0]*iy,jz) - expected));
        }
      }
    }
    
    cout << "Error dtA iterative solver rhs1: " << err << endl;
    if(gi.discr == discretization::fft)
      REQUIRE( err < 1e-12);
    else
      REQUIRE( err < 3e-3);

    // initialize E with something non-zero and check rhs
    componentwise_vec_omp(gi.N_xx, [&xx1, &xx2, &gi](Index idx, array<Index,2> i) {
      mfp<2> x  = gi.x(i);
      xx1(idx) = 1.0;
      xx2(idx) = sin(x[0])*sin(x[1]);
    });

    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      double z = gi.v(0, jz);
      vv1(jz) = 1.0;
      vv2(jz) = 0.5*sin(z);
    }
    X = {xx1.data(), xx2.data()};
    V = {vv1.data(), vv2.data()};
    initialize(E, X, V, ip_xx, ip_z, blas);

    dtA_it.compute_rhs(f, E, rho);
    dtA_full = dtA_it.rhs.full(blas);
    lr2<double> rhs_backup = dtA_it.rhs;

    err = 0.0;
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++) {
        for(Index ix=0;ix<gi.N_xx[0];ix++) {
          mfp<2> xy = gi.x({ix, iy});
          double z = gi.v(0, jz);
          double expected = -gi.C_A*cos(xy[0])*cos(xy[1])*sin(z)/(4.0*gi.M_e)
                            -gi.C_A/gi.M_e*(1+0.5*cos(xy[0])*cos(xy[1])*cos(z))*(1+0.5*sin(xy[0])*sin(xy[1])*sin(z));
          err = max(err, abs(dtA_full(ix+gi.N_xx[0]*iy,jz) - expected));
        }
      }
    }
    
    cout << "Error dtA iterative solver rhs2: " << err << endl;
    if(gi.discr == discretization::fft)
      REQUIRE( err < 1e-12);
    else
      REQUIRE( err < 3e-3);

    // check the lhs
    lr2<double> lhs = E;
    dtA_it.apply_lhs(lhs, rho);
    mat lhs_full = lhs.full(blas);

    err = 0.0;
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++) {
        for(Index ix=0;ix<gi.N_xx[0];ix++) {
          mfp<2> xy = gi.x({ix, iy});
          double z = gi.v(0, jz);
          double expected = sin(xy[0])*sin(xy[1])*sin(z)
                          + gi.C_A/gi.M_e*(1.0 + 0.5*cos(xy[0])*cos(xy[1])*cos(z))*(1.0 + 0.5*sin(xy[0])*sin(xy[1])*sin(z));
          err = max(err, abs(lhs_full(ix+gi.N_xx[0]*iy,jz) - expected));
        }
      }
    }

    cout << "Error dtA iterative solver lhs: " << err << endl;
    REQUIRE( err < 1e-12);


    // check CG solver
    lr2<double> dtA(gi.r, {gi.dxx_mult, gi.N_zv[0]});
    memset(dtA.S.data(), 0, dtA.rank()*dtA.rank()*sizeof(double));

    // setup the manufactured rhs
    componentwise_vec_omp(gi.N_xx, [&xx1, &xx2, &gi](Index idx, array<Index,2> i) {
      mfp<2> x  = gi.x(i);
      xx1(idx) = (2.0 + gi.C_A/gi.M_e)*sin(x[0])*sin(x[1]);
      xx2(idx) = 0.5*gi.C_A/gi.M_e*sin(x[0])*cos(x[0])*sin(x[1])*cos(x[1]);
    });

    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      double z = gi.v(0, jz);
      vv1(jz) = sin(z);
      vv2(jz) = cos(z)*sin(z);
    }

    lr2<double> manufactured_rhs(gi.r, {gi.dxx_mult, gi.N_zv[0]});
    X = {xx1.data(), xx2.data()};
    V = {vv1.data(), vv2.data()};
    initialize(manufactured_rhs, X, V, ip_xx, ip_z, blas);


    // setup rho = 1 + 0.5*cos(x)*cos(y)*cos(z)
    componentwise_vec_omp(gi.N_xx, [&xx1, &xx2, &gi](Index idx, array<Index,2> i) {
      mfp<2> x  = gi.x(i);
      xx1(idx) = 1.0;
      xx2(idx) = cos(x[0])*cos(x[1]);
    });

    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      double z = gi.v(0, jz);
      vv1(jz) = 1.0;
      vv2(jz) = 0.5*cos(z);
    }

    X = {xx1.data(), xx2.data()};
    V = {vv1.data(), vv2.data()};
    initialize(rho, X, V, ip_xx, ip_z, blas);


    // run the CG iteration
    dtA_it.operator()(dtA, f, E, rho, &manufactured_rhs);
    dtA_full = dtA.full(blas);

/*
    componentwise_vec_omp(gi.N_xx, [&xx1, &xx2, &gi](Index idx, array<Index,2> i) {
      mfp<2> x  = gi.x(i);
      xx1(idx) = sin(x[0])*sin(x[1]);
    });
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      double z = gi.v(0, jz);
      vv1(jz) = sin(z);
    }
    X = {xx1.data()};
    V = {vv1.data()};
    lr2<double> sol(gi.r, {gi.dxx_mult, gi.N_zv[0]});
    initialize(sol, X, V, ip_xx, ip_z, blas);

    lr2<double> Ap = sol;
    dtA_it.apply_lhs(Ap, rho);
    lr_add(1.0, manufactured_rhs, -1.0, Ap, dtA_it.medium, ip_xx, ip_z, blas);
    cout << "residual: " << lr_norm_sq(dtA_it.medium, blas) << endl;
    mat medium_full = dtA_it.medium.full(blas);
    mat Ap_full = Ap.full(blas);
    mat manufactured_rhs_full = manufactured_rhs.full(blas);
*/
    err = 0.0;
    //ofstream fs ("debug.txt");
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++) {
        for(Index ix=0;ix<gi.N_xx[0];ix++) {
          mfp<2> xy = gi.x({ix, iy});
          double z = gi.v(0, jz);
          double expected = sin(xy[0])*sin(xy[1])*sin(z);
          err = max(err, abs(dtA_full(ix+gi.N_xx[0]*iy,jz) - expected));

          //if(abs(dtA_full(ix+gi.N_xx[0]*iy,jz) - expected) > 0.05)
          //  cout << xy[0] << " " << xy[1] << " " << z << " " << dtA_full(ix+gi.N_xx[0]*iy,jz) << " " << expected << endl;

          /*        
          if(jz==gi.N_zv[0]/5 && iy==gi.N_xx[0]/5)
            fs << xy[0] << " " << expected << " " << dtA_full(ix+gi.N_xx[0]*iy,jz) << " " << medium_full(ix+gi.N_xx[0]*iy,jz) << " " << Ap_full(ix+gi.N_xx[0]*iy,jz) << " " << manufactured_rhs_full(ix+gi.N_xx[0]*iy,jz) << endl;
            */
        }
      }
    }

    //cout << dtA.S << endl;
    //cout << dtA.V << endl;
    //cout << dtA.X << endl;
    cout << "Error dtA iterative solver CG: " << err << endl;
    REQUIRE( err < 5e-8 );
  }


  /*
  *  advection in the z-direction
  */
  SECTION("advection_z") {
    gi.debug_adv_z = true;
    gi.debug_adv_v = false;
    gi.debug_adv_vA = false;

    double t_final = 0.1;


    lr2<double> f_final = integration("lie", t_final, 1e-3, gi, X, V); 
    mat f_full = f_final.full(blas);

    lr2<double> f_unconv_final = integration("unconventional", t_final, 1e-3, gi, X, V); 
    mat f_unconv_full = f_unconv_final.full(blas);
    
    double err_lie = 0.0, err_unconv = 0.0;
    for(Index jv=0;jv<gi.N_zv[1];jv++) {
      for(Index jz=0;jz<gi.N_zv[0];jz++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++) {
          for(Index ix=0;ix<gi.N_xx[0];ix++) {
            mfp<2> xy = gi.x({ix, iy});
            mfp<2> zv = gi.v({jz, jv});
            double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0]-t_final*zv[1])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1],2));
            double val = f_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            err_lie = max_err(err_lie, abs(val - expected));
            double val_unconv = f_unconv_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            err_unconv = max_err(err_unconv, abs(val_unconv - expected));
          }
        }
      }
    }
            
    cout << "Error advection z (Lie): " << err_lie << endl;
    cout << "Error advection z (Unconventional): " << err_unconv << endl;

    REQUIRE( err_lie <= 3e-5 );
    REQUIRE( err_unconv <= 3e-5 );
  }

  /*
  *  advection using the scalar potential in the v direction
  */
  SECTION("advection_v") {
    gi.debug_adv_z = false;
    gi.debug_adv_v = true;
    gi.debug_adv_vA = false;

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

    lr2<double> f_final = integration("lie", t_final, 1e-2, gi, X, V, &Kphi, &Vphi); 
    mat f_full = f_final.full(blas);
    
    lr2<double> f_final_unconv = integration("unconventional", t_final, 1e-2, gi, X, V, &Kphi, &Vphi); 
    mat f_unconv_full = f_final_unconv.full(blas);

    double err_lie = 0.0, err_unconv = 0.0;
    for(Index jv=0;jv<gi.N_zv[1];jv++) {
      for(Index jz=0;jz<gi.N_zv[0];jz++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++) {
          for(Index ix=0;ix<gi.N_xx[0];ix++) {
            mfp<2> xy = gi.x({ix, iy});
            mfp<2> zv = gi.v({jz, jv});
            double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1]-t_final/gi.M_e*cos(zv[0]),2));
            double val = f_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            err_lie = max_err(err_lie, abs(val - expected));
            double val_unconv = f_unconv_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            err_unconv = max_err(err_unconv, abs(val_unconv - expected));
          }
        }
      }
    }
            
    cout << "Error advection v (Lie): " << err_lie << endl;
    cout << "Error advection v (Unconventional): " << err_unconv << endl;

    REQUIRE( err_lie <= 5e-3 );
    REQUIRE( err_unconv <= 7e-3 );
  }

  /*
  *  advection using the scalar and magnetic potential in the v direction
  */
  SECTION("advection_v_dtA") {
    gi.debug_adv_z = false;
    gi.debug_adv_v = false;
    gi.debug_adv_vA = true;

    // 
    vec dtA_xx1({gi.dxx_mult});
    componentwise_vec_omp(gi.N_xx, [&dtA_xx1, &gi](Index idx, array<Index,2> i) {
      double x = gi.x(0, i[0]);
      double y = gi.x(1, i[1]);
      dtA_xx1(idx) = /*sin(x)* */ cos(x);
    });

    vec dtA_vv1({gi.N_zv[0]});
    for(Index i=0;i<gi.N_zv[0];i++) {
      double z = gi.v(0, i);
      dtA_vv1(i) = 1.0;
    }

    vector<const double*> dtA_X, dtA_V;
    dtA_X.push_back(dtA_xx1.begin());
    dtA_V.push_back(dtA_vv1.begin());
    lr2<double> dtA(gi.r, {gi.dxx_mult, gi.N_zv[0]});
    initialize(dtA, dtA_X, dtA_V, ip_xx, ip_z, blas);
    
    double t_final = 1e-2;


    lr2<double> f_unconv_final = integration("unconventional", t_final, 1e-4, gi, X, V, nullptr, nullptr, &dtA); 
    mat f_unconv_full = f_unconv_final.full(blas);

    lr2<double> f_final = integration("lie", t_final, 1e-4, gi, X, V, nullptr, nullptr, &dtA); 
    mat f_full = f_final.full(blas);
    

    double err_lie = 0.0, err_unconv = 0.0;
    for(Index jv=0;jv<gi.N_zv[1];jv++) {
      for(Index jz=0;jz<gi.N_zv[0];jz++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++) {
          for(Index ix=0;ix<gi.N_xx[0];ix++) {
            mfp<2> xy = gi.x({ix, iy});
            mfp<2> zv = gi.v({jz, jv});
            //double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1]-t_final/gi.M_e*sin(zv[0]) * sin(xy[0])*cos(xy[1]),2));
            double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1]-t_final/gi.M_e*cos(xy[0]),2));
            double val = f_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            err_lie = max_err(err_lie, abs(val - expected));
            double val_unconv = f_unconv_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            err_unconv = max_err(err_unconv, abs(val_unconv - expected));
          }
        }
      }
    }
            
    cout << "Error advection v (Lie): " << err_lie << endl;
    cout << "Error advection v (Unconventional): " << err_unconv << endl;

    REQUIRE( err_lie <= 6e-3 );
    REQUIRE( err_unconv <= 6e-3 );
  }


}
