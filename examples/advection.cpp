#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>

#include <random>
#include <complex>
#include <fftw3.h>
#include <cstring>

int main(){
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);

  Index Nx = 10; // NEEDS TO BE EVEN FOR FOURIER
  Index Nv = 10; // NEEDS TO BE EVEN FOR FOURIER

  int r = 1; // rank desired
  int n_b = 1; // number of actual basis functions

  double tau = 0.01;
  double tstar = 0.1;

  double ax = -M_PI;
  double bx = M_PI;

  double av = -M_PI;
  double bv = M_PI;

  int nsteps = tstar/tau;

  double hx = (bx-ax) / Nx;
  double hv = (bv-av) / Nv;

  vector<double*> X, V;

  vector<double> x,v, x1, v1;

  for(Index i = 0; i < Nx; i++){
    x.push_back(ax + i*hx);
    x1.push_back(cos(ax + i*hx));
  }

  for(Index i = 0; i < Nv; i++){
    v.push_back(av + i*hv);
    v1.push_back(exp(-pow((av + i*hv),2)/2.0));
  }

  X.push_back(x1.data());
  V.push_back(v1.data());

  vector<vector<double>> RX;
  vector<vector<double>> RV;

  for(int i = 0; i < (r-n_b); i++){
    vector<double> randx;
    vector<double> randv;
    for(Index j = 0; j < Nx; j++){
      randx.push_back(distribution(generator));
    }
    for(Index j = 0; j < Nv; j++){
      randv.push_back(distribution(generator));
    }
    RX.push_back(randx);
    RV.push_back(randv);

    X.push_back(RX[i].data());
    V.push_back(RV[i].data());
  }

  std::function<double(double*,double*)> ip_x = inner_product_from_const_weight(hx, Nx);
  std::function<double(double*,double*)> ip_v = inner_product_from_const_weight(hv, Nv);

  lr2<double> lr0(r,{Nx,Nv});

  initialize(lr0, X, V, n_b, ip_x, ip_v);

  lr2<double> lr_sol(r,{Nx,Nv});

  nsteps = 1;

  for(int i = 0; i < nsteps; i++){
    // K step

    matmul(lr0.X,lr0.S,lr_sol.X);

    multi_array<std::complex<double>,2> Khat({Nx/2+1,r});

    for(int j = 0; j<Nx; j++){
      for(int k = 0; k<r; k++){
        cout << lr_sol.X(j,k) << endl;
      }
    }

    // S step

    // L step
  }


  return 0;
}
