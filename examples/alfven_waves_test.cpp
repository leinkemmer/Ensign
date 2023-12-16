#include "alfven_waves.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


string discr_to_str(discretization discr) {
  return (discr==discretization::fft) ? "fft" : "lw";
}

struct test_config {

  test_config(Index r, discretization discr)
  : gi(r, {16, 16}, {64, 64}, {0.0,2*M_PI,0.0,2*M_PI}, {0.0,2*M_PI,-6.0/sqrt(0.1), 6.0/sqrt(0.1)}, 0.1, 3.0, 0.5, discr), cc(gi) {

    ip_xx = inner_product_from_const_weight(gi.h_xx[0]*gi.h_xx[1], gi.dxx_mult);
    ip_zv = inner_product_from_const_weight(gi.h_zv[0]*gi.h_zv[1], gi.dzv_mult);
    ip_z  = inner_product_from_const_weight(gi.h_zv[0], gi.N_zv[0]);

    xx1 = std::unique_ptr<vec>(new vec({gi.dxx_mult}));
    xx2 = std::unique_ptr<vec>(new vec({gi.dxx_mult}));
    componentwise_vec_omp(gi.N_xx, [this](Index idx, array<Index,2> i) {
      mfp<2> x  = gi.x(i);
      (*xx1)(idx) = 1.0;
      (*xx2)(idx) = cos(x[0])*cos(x[1]);
    });

    vv1 = std::unique_ptr<vec>(new vec({gi.dzv_mult}));
    vv2 = std::unique_ptr<vec>(new vec({gi.dzv_mult}));
    componentwise_vec_omp(gi.N_zv, [this](Index idx, array<Index,2> i) {
      mfp<2> zv  = gi.v(i);
      (*vv1)(idx) = sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
      (*vv2)(idx) = 0.5*cos(zv[0])*sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
    });

    X.push_back(xx1->begin());
    X.push_back(xx2->begin());
    V.push_back(vv1->begin());
    V.push_back(vv2->begin());
    f.resize(gi.r, {gi.dxx_mult, gi.dzv_mult});
    initialize(f, X, V, ip_xx, ip_zv, blas);
  }

  grid_info<2> gi;
  compute_coeff cc;
  std::function<double(double*,double*)> ip_xx;
  std::function<double(double*,double*)> ip_zv;
  std::function<double(double*,double*)> ip_z; 
  vector<const double*> X, V;
  lr2<double> f;
  std::unique_ptr<vec> xx1, xx2, vv1, vv2;
  blas_ops blas;
};


void test_scalar_potential(discretization discr) {
    test_config config(3, discr);
    grid_info<2> gi = config.gi;

    // scalar potential
    mat K({gi.dxx_mult,gi.r});
    config.blas.matmul(config.f.X, config.f.S, K);
    
    mat Kphi({gi.dxx_mult,gi.r});
    mat Vphi({gi.N_zv[0],gi.r});
    scalar_potential compute_phi(gi, config.blas);
    
    mat Kmrho({gi.dxx_mult,gi.r});
    compute_phi(K, config.f.V, Kphi, Vphi, &Kmrho);

    mat phi({gi.dxx_mult, gi.N_zv[0]});
    config.blas.matmul_transb(Kphi, Vphi, phi);

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
   
    cout << "Error scalar potential (" << discr_to_str(discr) << "): " << err << endl;
    REQUIRE( err < 1e-13 );
}

void test_vector_potential(discretization discr) {
    test_config config(3, discr);
    grid_info<2> gi = config.gi;

    // vector potential
    mat KA({gi.dxx_mult, gi.r});
    mat VA({gi.N_zv[0], gi.r});
    vector_potential compute_A(gi, config.blas);
    compute_A(config.f, KA, VA);

    mat A({gi.dxx_mult, gi.N_zv[0]});
    config.blas.matmul_transb(KA, VA, A);

    double err = 0.0;
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++) {
        for(Index ix=0;ix<gi.N_xx[0];ix++) {
          double expected = -gi.C_A*0.0;
          err = max(err, abs(A(ix+gi.N_xx[0]*iy,jz) - expected));
        }
      }
    }
    cout << "Error vector potential (" << discr_to_str(discr) << "): " << err << endl;
    REQUIRE( err < 1e-13 );
}

void test_dtA_iteration(discretization discr) {
    test_config config(5, discr);
    grid_info<2>& gi = config.gi;

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
    initialize(rho, X, V, config.ip_xx, config.ip_z, config.blas);

    // set E to zero
    lr2<double> E(gi.r, {gi.dxx_mult, gi.N_zv[0]});
    memset(E.S.data(), 0, E.rank()*E.rank()*sizeof(double));

    // check the rhs
    dtA_iterative_solver dtA_it(gi, config.blas);
    dtA_it.compute_rhs(config.f, E, rho);
    mat dtA_full = dtA_it.rhs.full(config.blas);

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
    
    cout << "Error dtA iterative solver rhs1 (" << discr_to_str(discr) << "): " << err << endl;
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
    initialize(E, X, V, config.ip_xx, config.ip_z, config.blas);

    dtA_it.compute_rhs(config.f, E, rho);
    dtA_full = dtA_it.rhs.full(config.blas);
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
    
    cout << "Error dtA iterative solver rhs2 (" << discr_to_str(discr) << "): " << err << endl;
    if(gi.discr == discretization::fft)
      REQUIRE( err < 1e-12);
    else
      REQUIRE( err < 3e-3);

    // check the lhs
    lr2<double> lhs = E;
    dtA_it.apply_lhs(lhs, rho);
    mat lhs_full = lhs.full(config.blas);

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

    cout << "Error dtA iterative solver lhs (" << discr_to_str(discr) << "): " << err << endl;
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
    initialize(manufactured_rhs, X, V, config.ip_xx, config.ip_z, config.blas);


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
    initialize(rho, X, V, config.ip_xx, config.ip_z, config.blas);


    // run the CG iteration
    dtA_it.operator()(dtA, config.f, E, rho, &manufactured_rhs);
    dtA_full = dtA.full(config.blas);

    err = 0.0;
    //ofstream fs ("debug.txt");
    for(Index jz=0;jz<gi.N_zv[0];jz++) {
      for(Index iy=0;iy<gi.N_xx[1];iy++) {
        for(Index ix=0;ix<gi.N_xx[0];ix++) {
          mfp<2> xy = gi.x({ix, iy});
          double z = gi.v(0, jz);
          double expected = sin(xy[0])*sin(xy[1])*sin(z);
          err = max(err, abs(dtA_full(ix+gi.N_xx[0]*iy,jz) - expected));
        }
      }
    }

    cout << "Error dtA iterative solver CG (" << discr_to_str(discr) << "): " << err << endl;
    REQUIRE( err < 5e-8 );
}

void test_advection_z(string method, discretization discr) {
    test_config config(3, discr);
    grid_info<2> gi = config.gi;

    gi.debug_adv_z = true;
    gi.debug_adv_v = false;
    gi.debug_adv_vA = false;

    double t_final = 0.1;

    integrator integrate(method, t_final, 1e-3, false, 0, gi, config.X, config.V);
    lr2<double> f_final = integrate.run();
    mat f_full = f_final.full(config.blas);
    
    double err = 0.0;
    for(Index jv=0;jv<gi.N_zv[1];jv++) {
      for(Index jz=0;jz<gi.N_zv[0];jz++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++) {
          for(Index ix=0;ix<gi.N_xx[0];ix++) {
            mfp<2> xy = gi.x({ix, iy});
            mfp<2> zv = gi.v({jz, jv});
            double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0]-t_final*zv[1])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1],2));
            double val = f_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            err = max_err(err, abs(val - expected));
          }
        }
      }
    }
            
    cout << "Error advection z (" << method << ", " << discr_to_str(discr) <<  "): " << err << endl;
    REQUIRE( err <= 3e-5 );
}

void test_advection_v(string method, discretization discr) {
    test_config config(3, discr);
    grid_info<2> gi = config.gi;

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

    integrator integrate(method, t_final, 1e-2, false, 0, gi, config.X, config.V, &Kphi, &Vphi);
    lr2<double> f_final = integrate.run();
    mat f_full = f_final.full(config.blas);

    double err = 0.0;
    for(Index jv=0;jv<gi.N_zv[1];jv++) {
      for(Index jz=0;jz<gi.N_zv[0];jz++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++) {
          for(Index ix=0;ix<gi.N_xx[0];ix++) {
            mfp<2> xy = gi.x({ix, iy});
            mfp<2> zv = gi.v({jz, jv});
            double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1]-t_final/gi.M_e*cos(zv[0]),2));
            double val = f_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            err = max_err(err, abs(val - expected));
          }
        }
      }
    }
            
    cout << "Error advection v (" << method << ", " << discr_to_str(discr) << "): " << err << endl;

    if(method == "lie" || method == "augmented")
      REQUIRE( err <= 5e-3 );
    else
      REQUIRE( err <= 7e-3 );
}

void test_advection_v_dtA(string method, discretization discr) {
    test_config config(3, discr);
    grid_info<2> gi = config.gi;

    gi.debug_adv_z = false;
    gi.debug_adv_v = false;
    gi.debug_adv_vA = true;

    // 
    vec dtA_xx1({gi.dxx_mult});
    componentwise_vec_omp(gi.N_xx, [&dtA_xx1, &gi](Index idx, array<Index,2> i) {
      double x = gi.x(0, i[0]);
      dtA_xx1(idx) = cos(x);
    });

    vec dtA_vv1({gi.N_zv[0]});
    for(Index i=0;i<gi.N_zv[0];i++) {
      dtA_vv1(i) = 1.0;
    }

    vector<const double*> dtA_X, dtA_V;
    dtA_X.push_back(dtA_xx1.begin());
    dtA_V.push_back(dtA_vv1.begin());
    lr2<double> dtA(gi.r, {gi.dxx_mult, gi.N_zv[0]});
    initialize(dtA, dtA_X, dtA_V, config.ip_xx, config.ip_z, config.blas);
    
    double t_final = 1e-2;

    integrator integration(method, t_final, 1e-4, false, 0, gi, config.X, config.V, nullptr, nullptr, &dtA); 
    lr2<double> f_final = integration.run();
    mat f_full = f_final.full(config.blas);

    double err = 0.0;
    for(Index jv=0;jv<gi.N_zv[1];jv++) {
      for(Index jz=0;jz<gi.N_zv[0];jz++) {
        for(Index iy=0;iy<gi.N_xx[1];iy++) {
          for(Index ix=0;ix<gi.N_xx[0];ix++) {
            mfp<2> xy = gi.x({ix, iy});
            mfp<2> zv = gi.v({jz, jv});
            double expected = sqrt(gi.M_e/M_PI)*(1.0+0.5*cos(zv[0])*cos(xy[0])*cos(xy[1]))*exp(-gi.M_e*pow(zv[1]-t_final/gi.M_e*cos(xy[0]),2));
            double val = f_full(ix+gi.N_xx[0]*iy,jz+gi.N_zv[0]*jv);
            err = max_err(err, abs(val - expected));
          }
        }
      }
    }
            
    cout << "Error advection v/dtA (" << method << ", " << discr_to_str(discr) << "): " << err << endl;

    REQUIRE( err <= 4e-3 );
}


TEST_CASE( "Alfven waves", "[alfven_waves]" ) {

  #ifdef __OPENMP
  omp_set_num_threads(1);
  #endif

  /*
  *  initialization
  */
  SECTION("INITIALIZE") {
    test_config config(3, discretization::fft);
    grid_info<2> gi = config.gi;

    mat K({gi.dxx_mult,gi.r});
    config.blas.matmul(config.f.X, config.f.S, K);

    mat f_full({gi.dxx_mult, gi.dzv_mult});
    config.blas.matmul_transb(K, config.f.V, f_full);

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
    test_scalar_potential(discretization::fft);
    test_scalar_potential(discretization::lw);
  }


  /*
  *  Computation of the vector potential
  */ 
  SECTION("vector_potential") {
    test_vector_potential(discretization::fft);
    test_vector_potential(discretization::lw);
  }


  /*
  *  Iterative solver for the vector potential
  */
  SECTION("dtA_iterative_solver") {
    test_dtA_iteration(discretization::fft);
    test_dtA_iteration(discretization::lw);
  }


  /*
  *  advection in the z-direction
  */
  SECTION("advection_z") {
    test_advection_z("lie", discretization::fft);
    test_advection_z("lie", discretization::lw);
    test_advection_z("unconventional", discretization::fft);
    test_advection_z("unconventional", discretization::lw);
    test_advection_z("augmented", discretization::fft);
    test_advection_z("augmented", discretization::lw);
    test_advection_z("strang", discretization::fft);
    test_advection_z("strang", discretization::lw);
  }

  /*
  *  advection using the scalar potential in the v direction
  */
  SECTION("advection_v") {
    test_advection_v("lie", discretization::fft);
    test_advection_v("lie", discretization::lw);
    test_advection_v("unconventional", discretization::fft);
    test_advection_v("unconventional", discretization::lw);
    test_advection_v("augmented", discretization::fft);
    test_advection_v("augmented", discretization::lw);
    test_advection_v("strang", discretization::fft);
    test_advection_v("strang", discretization::lw);
  }

  /*
  *  advection using the scalar and magnetic potential in the v direction
  */
  SECTION("advection_v_dtA") {
    test_advection_v_dtA("lie", discretization::fft);
    test_advection_v_dtA("lie", discretization::lw);
    test_advection_v_dtA("unconventional", discretization::fft);
    test_advection_v_dtA("unconventional", discretization::lw);
    test_advection_v_dtA("augmented", discretization::fft);
    test_advection_v_dtA("augmented", discretization::lw);
    test_advection_v_dtA("strang", discretization::fft);
    test_advection_v_dtA("strang", discretization::lw);
  }

}
