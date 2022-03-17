#include "alfven_waves.hpp"

int main() {

  double rho_i = 1e-3;
  double kperp = 0.2/rho_i;
  double kx = sqrt(0.5)*kperp;
  double ky = sqrt(0.5)*kperp;
  double M_e = 1.0/1830.0;
  double beta = 0.00219;
  double C_P = 1.0/pow(rho_i,2);
  double C_A = beta/pow(rho_i,2);
  double kpar = 1.0;
  double Vmax = 6.0/sqrt(M_e);
  double alpha = 1e-3;

  Index r = 5;
  mind<2> N_xx = {100, 100};
  mind<2> N_zv = {100, 100};
  mfp<4> lim_xx = {0.0,2*M_PI/kx,0.0,2*M_PI/ky};
  mfp<4> lim_zv = {0.0,2*M_PI,-Vmax,Vmax};
  grid_info<2> gi(r, N_xx, N_zv, lim_xx, lim_zv, M_e, C_P, C_A); 


  vec xx1({gi.dxx_mult}), xx2({gi.dxx_mult});
  componentwise_vec_omp(gi.N_xx, [&xx1, &xx2, &gi, kx, ky](Index idx, array<Index,2> i) {
    mfp<2> x  = gi.x(i);
    xx1(idx) = 1.0;
    xx2(idx) = cos(kx*x[0])*cos(ky*x[1]);
  });

  vec vv1({gi.dzv_mult}), vv2({gi.dzv_mult});
  componentwise_vec_omp(gi.N_zv, [&vv1, &vv2, &gi, &alpha](Index idx, array<Index,2> i) {
    mfp<2> zv  = gi.v(i);
    vv1(idx) = sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
    vv2(idx) = alpha*cos(zv[0])*sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
  });
  vector<const double*> X, V;
  X.push_back(xx1.begin());
  X.push_back(xx2.begin());
  V.push_back(vv1.begin());
  V.push_back(vv2.begin());

  integration(20.0, 5e-5, gi, X, V);
  //integration(0.001, 5e-5, gi, X, V);
}

