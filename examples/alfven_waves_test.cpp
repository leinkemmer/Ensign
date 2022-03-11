#include "alfven_waves.hpp"

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
  grid_info<2> gi(r, N_xx, N_zv, lim_xx, lim_zv, M_e, C_P, 0.0); 

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