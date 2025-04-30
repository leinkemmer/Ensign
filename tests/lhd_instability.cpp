#include "../examples/lhd_instability.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

mfp<3> a = {0.0, -1.125, -1.128};
mfp<3> b = {1.414, 1.125, 1.122};
double g = 0.5;
double B = 0.8;
double Omega = 2.5;
double charge = -0.5;
double mass = 1.0/25.0;
double alpha = 2e-4;
double T_e = 6.25e-5;
double T_i = T_e;

double X1(double y) {
    return 0.838865366424064*(1.0 + 0.1*cos(2.0*M_PI/1.414*y));
}

double X2(double y) {
    return 1.1892969170906877*sin(2.0*M_PI/1.414*y);
}

double V1(mfp<2> v) {
    return exp(-20.0*pow(v[0]-0.2,2))*exp(-22.0*pow(v[1],2));
}

double V2(mfp<2> v) {
    return exp(-20.0*pow(v[0],2))*exp(-22.0*pow(v[1]+0.2,2));
}

struct test_config {

  test_config(Index r)
  : gi(r, 128, {129,130}, a, b, g, B, charge, mass, Omega) {

    Xref.resize({gi.n_x, r});
    Xref.set_zero();

    // setup initial condition
    xx1 = std::unique_ptr<vec>(new vec({gi.n_x}));
    xx2 = std::unique_ptr<vec>(new vec({gi.n_x}));
    for(Index i=0;i<gi.n_x;i++) {
      double y  = gi.x(i);
      (*xx1)(i) = X1(y);
      (*xx2)(i) = X2(y);
      Xref(i,0) = X1(y);
      Xref(i,1) = X2(y);
    }

    vv1 = std::unique_ptr<vec>(new vec({gi.N_v}));
    vv2 = std::unique_ptr<vec>(new vec({gi.N_v}));
    componentwise_vec_omp(gi.n_v, [this](Index idx, array<Index,2> i) {
      mfp<2> v  = gi.v(i);
      (*vv1)(idx) = V1(v); 
      (*vv2)(idx) = V2(v); 
    });

    X.push_back(xx1->begin());
    X.push_back(xx2->begin());

    V.push_back(vv1->begin());
    V.push_back(vv2->begin());
  }

  grid_info gi;
  vector<const double*> X, V;
  mat Xref;
  std::unique_ptr<vec> xx1, xx2, vv1, vv2;
  blas_ops blas;
};


// Even if we supply an orthogonal set of basis functions the orthogonalization routine
// can flip the sign. This is not the case for gram_schmidt but many linear algebra libraries
// do that. The code here is to figure out the correct sign.
// This is required because e.g. for the term g \partial_{vx} L there is no coefficient and so
// this does not cancel out as in th other cases.
vec get_conv(const mat& Xref, const vlasov& vl, const grid_info& gi, blas_ops& blas) {
  mat M({gi.r, gi.r});
  coeff(Xref, vl.f.X, gi.h_x, M, blas);
  
  vec conv({gi.r});
  conv.set_zero();
  conv(0) = -1.0 + 2.0*(M(0,0) >= 0);
  conv(1) = -1.0 + 2.0*(M(1,1) >= 0);
  return conv;
}


TEST_CASE( "LHD instability", "[lhd_instability]" ) {


 SECTION("INITIALIZATION") {
  test_config tc(3);
  grid_info& gi = tc.gi;
  vlasov vl(tc.gi, tc.X, tc.V);

  mat K({gi.n_x, gi.r});
  K.set_zero();
  tc.blas.matmul(vl.f.X, vl.f.S, K);

  mat f_full({gi.n_x, gi.N_v});
  tc.blas.matmul_transb(K, vl.f.V, f_full);

  double err=0.0;
  for(Index ivy=0;ivy<gi.n_v[1];ivy++) {
    for(Index ivx=0;ivx<gi.n_v[0];ivx++) {
      for(Index iy=0;iy<gi.n_x;iy++) {
        double y = gi.x(iy);
        mfp<2> v = gi.v({ivx,ivy});
        double exact = X1(y)*V1(v) + X2(y)*V2(v);
        double f_val = f_full(iy, gi.lin_idx_v({ivx,ivy}));
        err = max(err, abs(f_val-exact));
      }
    }
  }
  cout << "ERROR_INIT: " << err << endl;
  REQUIRE(err <= 1e-14); 
 }


 SECTION("RHS K") {
  test_config tc(3);
  grid_info& gi = tc.gi;
  vlasov vl(tc.gi, tc.X, tc.V);

  mat K({gi.n_x, gi.r});
  mat Kout = K;
  vec E({gi.n_x});
  E.set_zero();
  tc.blas.matmul(vl.f.X, vl.f.S, K);

  for(Index i=0;i<gi.n_x;i++) {
    double y = gi.x(i);
    E(i) = 100.0*pow(cos(4.443553965473541*y),2);
  }

  mat V({gi.N_v, gi.r});
  for(Index j=0;j<gi.n_v[0];j++) {
    for(Index i=0;i<gi.n_v[0];i++) {
      mfp<2> v  = gi.v({i,j});
      V(gi.lin_idx_v({i,j}), 0) = V1(v);
      V(gi.lin_idx_v({i,j}), 1) = V2(v); 
    }
  }

  vl.compute_C1(V,vl.f.V);
  vl.compute_C2(V,vl.f.V);
  vl.compute_C3(V,vl.f.V);
  vl.compute_C4(V,vl.f.V);
  vl.compute_C5(V,vl.f.V);
  vl.rhs_K(K, Kout, E);

  double err1 = 0.0, m1 = 0.0, err2 = 0.0, m2 = 0.0;
  for(Index i=0;i<gi.n_x;i++) {
    double y = gi.x(i);
    // first term
    double exact1 = 0.01708468303743943*cos(4.443553965473541*y);
    double exact2 = 0.07914879119339357*cos(4.443553965473541*y)-0.001205060627879167*sin(4.443553965473541*y);

    // second term
    exact1 += g*0.153792960951415*sin(4.443553965473541*y);
    exact2 += g*(-0.1084771907569935-0.01084771907569934*cos(4.443553965473541*y));

    // third term
    double Ehat = 100.0*pow(cos(4.443553965473541*y),2);
    exact1 += Ehat*0.1691722570465565*sin(4.443553965473541*y);
    exact2 += Ehat*(-0.1193249098326925-0.01193249098326925*cos(4.443553965473541*y));

    // third term
    double Bhat = charge*Omega*B/mass;
    exact1 += Bhat*(-0.03229652179979716)*sin(4.443553965473541*y);
    exact2 += Bhat*(0.02278021005896862+0.002278021005896863*cos(4.443553965473541*y));
    
    err1 = max(err1, abs(Kout(i,0)-exact1));
    err2 = max(err2, abs(Kout(i,1)-exact2));
    m1 = max(m1, abs(exact1));
    m2 = max(m2, abs(exact2));
  }


  cout << "ERROR RHS_K: "<< err1/m1 <<  " " << err2/m2 << endl;
  REQUIRE( err1/m1 <= 3e-3);
  REQUIRE( err2/m2 <= 3e-3);
 }

 SECTION("RHS L") {
  test_config tc(3);
  grid_info& gi = tc.gi;
  vlasov vl(tc.gi, tc.X, tc.V);

  vec conv = get_conv(tc.Xref, vl, gi, tc.blas);

  mat L({gi.N_v, gi.r});
  mat Lout = L;
  tc.blas.matmul_transb(vl.f.V, vl.f.S, L);

  vec Ehat({gi.n_x});
  Ehat.set_zero();
  for(Index i=0;i<gi.n_x;i++) {
    double y = gi.x(i);
    Ehat(i) = pow(cos(4.443553965473541*y),2);
  }

  mat X({gi.n_x, gi.r});
  for(Index i=0;i<gi.n_x;i++) {
      double x  = gi.x(i);
      X(i, 0) = X1(x);
      X(i, 1) = X2(x);
  }

  vl.compute_D1(X,vl.f.X);
  vl.compute_D2(X,vl.f.X,Ehat);
  vl.rhs_L(L, Lout);

  double err1 = 0.0, m1 = 0.0, err2 = 0.0, m2 = 0.0;
  for(Index j=0;j<gi.n_v[1];j++) {
    for(Index i=0;i<gi.n_v[0];i++) {
      mfp<2> v = gi.v({i,j});
      double vx = v[0]; double vy = v[1];
      // first term
      double exact1 = -0.3134241308377849*exp(-20.0*pow(vx,2)-22.0*pow(0.2+vy,2))*vy;
      double exact2 =  0.31342413083778514*exp(-20.0*pow(-0.2+vx,2)-22.0*pow(vy,2))*vy;
      // second term
      exact1 += conv(0)*g*exp((8.-20.*vx)*vx-22.*pow(vy,2))*(-3.5946317129377743 + 17.973158564688866*vx);
      exact2 += conv(0)*g*16.591316467263255*exp(-20.*pow(vx,2)+(-8.8-22.*vy)*vy)*vx;
      // third term (E is already integrated out here)
      exact1 += 22.054726368159216*exp(-20.*pow(-0.2+vx,2)-22.*pow(vy,2))*vy;
      exact2 += exp(-20.*pow(vx,2)+(-8.8-22.*vy)*vy)*(0.912522405699479+4.562612028497394*vy);
      // fourth term
      double Bhat = charge*Omega*B/mass;
      exact1 += conv(0)*Bhat*exp((8.-20.*vx)*vx-22.*pow(vy,2))*(-3.594631712937776-1.797315856468888*vx)*vy;
      exact2 += conv(1)*Bhat*exp(-20.*pow(vx,2)+(-8.8-22.*vy)*vy)*vx*(-3.650089622797918-1.6591316467263262*vy);

      err1 = max(err1, abs(Lout(gi.lin_idx_v({i,j}),0)-exact1));
      err2 = max(err2, abs(Lout(gi.lin_idx_v({i,j}),1)-exact2));
      m1 = max(m1, abs(exact1));
      m2 = max(m2, abs(exact2));
    }
  }

  cout << "ERROR RHS_L: " << err1/m1 << " " << err2/m2 << endl;

  REQUIRE( err1/m1 <= 6e-3 );
  REQUIRE( err2/m2 <= 6e-3 );
 }


 SECTION("RHS S") {
  test_config tc(2);
  grid_info& gi = tc.gi;
  vlasov vl(tc.gi, tc.X, tc.V);
  
  vec conv = get_conv(tc.Xref, vl, gi, tc.blas);

  vec Ehat({gi.n_x});
  Ehat.set_zero();
  for(Index i=0;i<gi.n_x;i++) {
    double y = gi.x(i);
    Ehat(i) = pow(cos(4.443553965473541*y),2);
  }

  mat X({gi.n_x, gi.r});
  for(Index i=0;i<gi.n_x;i++) {
      double x  = gi.x(i);
      X(i, 0) = X1(x);
      X(i, 1) = X2(x);
  }
  
  mat V({gi.N_v, gi.r});
  for(Index j=0;j<gi.n_v[0];j++) {
    for(Index i=0;i<gi.n_v[0];i++) {
      mfp<2> v  = gi.v({i,j});
      V(gi.lin_idx_v({i,j}), 0) = V1(v);
      V(gi.lin_idx_v({i,j}), 1) = V2(v); 
    }
  }

  vl.compute_D1(X,vl.f.X);
  vl.compute_D2(X,vl.f.X,Ehat);

  vl.compute_C1(V,vl.f.V);
  vl.compute_C2(V,vl.f.V);
  vl.compute_C3(V,vl.f.V);
  vl.compute_C4(V,vl.f.V);
  vl.compute_C5(V,vl.f.V);

  mat Sout({gi.r, gi.r});
  vl.rhs_S(vl.f.S, Sout);

  mat Sexact({gi.r, gi.r});
  double Bhat = charge*Omega*B/mass;
  Sexact(0,0) = 0.001013254646978351 + conv(0)*g*0 + 0;
  Sexact(0,1) = 0.004694139206661268 - conv(0)*g*0.1293141844911448 - 0.07129972386184644 + conv(0)*Bhat*0.02715597874314041;
  Sexact(1,0) = 0.0 + conv(1)*g*0.12931418449114485 + 0.03556140073506485 - conv(1)*Bhat*0.02715597874314043;
  Sexact(1,1) = -0.0010132546469783514 + conv(1)*g*0 + 0;

  cout << "ERROR RHS_S: ";
  for(Index i=0;i<2;i++) {
    for(Index j=0;j<2;j++) {
      cout << Sexact(i,j) << " " << Sout(i,j) << endl;
      double err = abs(Sexact(i,j)-Sout(i,j));
      cout << err <<  " ";
      REQUIRE( err <= 3.5e-3 ); 
    }
  }
  cout << endl;

}


 SECTION("MOMENTS") {
  test_config tc(3);
  grid_info& gi = tc.gi;
  vlasov vl(tc.gi, tc.X, tc.V);

  vec n({gi.n_x}), hx({gi.n_x}), hy({gi.n_x});
  vl.compute_nh(n, hx, hy); 
  
  double errn=0.0, mn=0.0, errhx=0.0, mhx=0.0, errhy=0.0, mhy=0.0;
  for(Index i=0;i<gi.n_x;i++) {
    double y = gi.x(i);

    // n
    double exactn = 0.1256364908933308+0.01256364908933308*cos(4.443553965473541*y)+0.1781204679378335*sin(4.443553965473541*y);
    errn = max(errn, abs(n(i)-exactn));
    mn = max(mn, abs(exactn));

    // hx
    double exacthx = 0.02512729788548804+0.002512729788548804*cos(4.443553965473541*y);
    errhx = max(errhx, abs(hx(i)-exacthx));
    mhx = max(mhx, abs(exacthx));
    
    // hy
    double exacthy = -0.03562409352421863*sin(4.443553965473541*y);
    errhy = max(errhy, abs(hy(i)-exacthy));
    mhy = max(mhy, abs(exacthy));
  }
  cout << "ERROR n hx hy: " << errn/mn << " " << errhx/mhx << " "<< errhy/mhy << endl;
  REQUIRE( errn/mn <= 1e-8 );
  REQUIRE( errhx/mhx <= 3e-8 );
  REQUIRE( errhy/mhy <= 1e-8 );

 }


 SECTION("POISSON_AND_NU") {
  test_config tc(3);
  grid_info gi_e = tc.gi;
  grid_info gi_i = tc.gi;
  gi_i.q *= -1.0;

  vec n_e({gi_e.n_x}), n_i({gi_i.n_x}), hy_e({gi_e.n_x}), hy_i({gi_i.n_x});
  for(Index i=0;i<gi_e.n_x;i++) {
    double y = gi_e.x(i);
    n_e(i) = 0.2 + 0.1*cos(2.0*M_PI/1.414*y);
    n_i(i) = 0.2 + 0.1*sin(2.0*M_PI/1.414*y);
    hy_e(i) = 0.1 + 0.05*sin(4.0*M_PI/1.414*y);
    hy_i(i) = 0.1 + 0.05*cos(4.0*M_PI/1.414*y);
  }

  poisson po(gi_e, gi_i);
  po.compute(n_e, n_i);

  double err = 0.0, m = 0.0;
  for(Index i=0;i<gi_e.n_x;i++) {
     double y = gi_e.x(i);
     double exact = abs(charge)*(-0.022504508953194*cos(4.443553965473541*y)-0.022504508953194*sin(4.443553965473541*y));
     double val = po.E(i);
     err = max(err, abs(val-exact));
     m = max(m, abs(exact));
  }

  double ee_exact = pow(charge,2)*0.0003580622167196429;
  cout << "ERROR Poisson: " << err/m << " " << abs(po.ee-ee_exact)/ee_exact << endl;
  REQUIRE( err/m <= 1e-14 );
  REQUIRE( abs(po.ee-ee_exact)/ee_exact <= 1e-14 );

  // check nu
  double nu_exact = abs(charge)*0.0993588362775873*charge/mass;
  double nu = po.compute_anomcoll(n_e, n_i, hy_e, hy_i);
  double err_nu = abs(nu - nu_exact)/abs(nu_exact); 
  cout << "ERROR nu: " << err_nu << endl;
  REQUIRE( err_nu <= 1e-14 );
 }

}
