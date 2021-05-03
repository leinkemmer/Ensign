#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>

#include <random>
#include <complex>
#include <fftw3.h>
#include <cstring>

extern "C" {
extern int dgees_(char*,char*,void*,int*,double*,int*, int*, double*, double*, double*, int*, double*, int*, bool*,int*);
}

int main(){

  Index Nx = 10; // NEEDS TO BE EVEN FOR FOURIER
  Index Nv = 10; // NEEDS TO BE EVEN FOR FOURIER

  int r = 2; // rank desired
  int n_b = 1; // number of actual basis functions

  double tstar = 0.1; // final time
  double tau = 0.01; // time step splitting

  int nsteps_ee = 1000; // number of time steps for explicit euler

  double ax = -M_PI;
  double bx = M_PI;

  double av = -M_PI;
  double bv = M_PI;

  int nsteps = tstar/tau;

  double ts_ee_forward = tau/nsteps_ee;
  double ts_ee_backward = -tau/nsteps_ee;

  double hx = (bx-ax) / Nx;
  double hv = (bv-av) / Nv;

  vector<double*> X, V;

  vector<double> x, v, x1, v1;

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
/*
  vector<vector<double>> RX;
  vector<vector<double>> RV;

  std::default_random_engine generator(time(0));
  std::normal_distribution<double> distribution(0.0,1.0);

  for(Index i = 0; i < (r-n_b); i++){
    vector<double> randx;
    vector<double> randv;
    for(Index j = 0; j < Nx; j++){
      randx.push_back(distribution(generator));
      //randx.push_back(j);
    }
    for(Index j = 0; j < Nv; j++){
      randv.push_back(distribution(generator));
      //randv.push_back(j);
    }
    RX.push_back(randx);
    RV.push_back(randv);

    X.push_back(RX[i].data());
    V.push_back(RV[i].data());
  }
*/
  std::function<double(double*,double*)> ip_x = inner_product_from_const_weight(hx, Nx);
  std::function<double(double*,double*)> ip_v = inner_product_from_const_weight(hv, Nv);

  lr2<double> lr0(r,{Nx,Nv});

  initialize(lr0, X, V, ip_x, ip_v);

  lr2<double> lr_sol(r,{Nx,Nv});

  // For FFT -- Pay attention we have to cast to int as Index seems not to work with fftw_many

  int rank = 1;
  int n = int(Nx);
  int howmany = r;
  int istride = 1;
  int ostride = 1;
  int* inembed = &n;
  int* onembed = &n;
  Index idist = Nx;
  Index odist = Nx/2 + 1;

  multi_array<complex<double>,2> Khat({Nx/2+1,r});

  fftw_plan p;
  p = fftw_plan_many_dft_r2c(rank, &n, howmany, lr_sol.X.begin(), inembed, istride, idist, (fftw_complex*)Khat.begin(), onembed, ostride, odist, FFTW_MEASURE);

  fftw_plan q;
  idist = Nx/2 + 1;
  odist = Nx;

  q = fftw_plan_many_dft_c2r(rank, &n, howmany, (fftw_complex*) Khat.begin(), inembed, istride, idist, lr_sol.X.begin(), onembed, ostride, odist, FFTW_MEASURE);

  // Mandatory to initialize after plan creation if we use FFTW_MEASURE
  lr_sol.X = lr0.X;
  lr_sol.S = lr0.S;
  lr_sol.V = lr0.V;

  vector<complex<double>> lambda;
  for(int j = 0; j < (Nx/2 + 1) ; j++){
    lambda.push_back(complex<double>(0.0,2.0*M_PI/(bx-ax)*j));
  }

  double nc_backward = 1.0 / Nx;

  // For C coefficients
  multi_array<double,2> C({r,r});

  vector<double> w;
  for(int j = 0; j < Nv; j++){
    w.push_back(v[j] * hv);
  }

  // For Schur decomposition
  vector<double> dc_r,dc_i;
  dc_r.reserve(r);
  dc_i.reserve(r);

  multi_array<double,2> T({r,r});

  multi_array<complex<double>,2> Mhat({Nx/2 + 1,r});
  multi_array<complex<double>,2> Tc({r,r});

  int value = 0;
  char jobvs = 'V';
  char sort = 'N';
  int nn = r;
  int lda = r;
  int ldvs = r;
  int info;
  int lwork = -1;
  double work_opt;

  // Dumb call to obtain optimal value to work
  dgees_(&jobvs,&sort,nullptr,&nn,T.begin(),&lda,&value,dc_r.data(),dc_i.data(),T.begin(),&ldvs,&work_opt,&lwork,nullptr,&info);

  lwork = int(work_opt);
  vector<double> work;
  work.reserve(lwork);

  // For D coefficients

  multi_array<double,2> D({r,r});

  vector<double> ww;
  for(int j = 0; j< Nx; j++){
    ww.push_back(hx);
  }


  for(int i = 0; i < nsteps; i++){

    /* K step */
    multi_array<double,2> tmpX(lr_sol.X);

    matmul(tmpX,lr_sol.S,lr_sol.X);

    fftw_execute_dft_r2c(p,lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    coeff(lr_sol.V, lr_sol.V, w.data(), C);

    multi_array<double,2> D_C(C); // needed because dgees overwrites input, and I need C later on

    dgees_(&jobvs,&sort,nullptr,&nn,D_C.begin(),&lda,&value,dc_r.data(),dc_i.data(),T.begin(),&ldvs,work.data(),&lwork,nullptr,&info);

    // Forced casting as we need two complex matrices
    for(int j = 0; j < r; j++){
      for(int k = 0; k < r; k++){
        Tc(j,k) = complex<double>(T(j,k),0.0);
      }
    }

    matmul(Khat,Tc,Mhat);

    for(int j = 0; j < (Nx/2 + 1); j++){
      for(int k = 0; k < r; k++){
        Mhat(j,k) *= exp(-tau*lambda[j]*dc_r[k])*nc_backward;
      }
    }

    matmul_transb(Mhat,Tc,Khat);

    fftw_execute_dft_c2r(q,(fftw_complex*)Khat.begin(),lr_sol.X.begin());

    gram_schmidt(lr_sol.X, lr_sol.S, ip_x);

    /* S step */

    multi_array<double,2> dX({Nx,r});

    fftw_execute_dft_r2c(p,lr_sol.X.begin(),(fftw_complex*)Khat.begin());


    for(int k = 0; k<r; k++){
        for(int j = 0; j<(Nx/2 + 1); j++){
            Khat(j,k) *= lambda[j]*nc_backward;
        }
    }

    fftw_execute_dft_c2r(q,(fftw_complex*)Khat.begin(),dX.begin());

    coeff(lr_sol.X, dX, ww.data(), D);

    // Explicit Euler
    for(int j = 0; j< nsteps_ee; j++){
      matmul_transb(lr_sol.S,C,D_C);
      matmul(D,D_C,T);
      T *= ts_ee_forward;
      lr_sol.S += T;
    }

    /* L step */

    multi_array<double,2> tmpV(lr_sol.V);
    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    // Explicit euler
    for(int j = 0; j< nsteps_ee; j++){
      matmul_transb(lr_sol.V,D,tmpV);

      for(int k = 0; k<Nv; k++){
        for(int l = 0; l<r; l++){
          tmpV(k,l) *= (ts_ee_backward*v[k]);
        }
      }

      lr_sol.V += tmpV;
    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_v);

    transpose_inplace(lr_sol.S);

  }

  multi_array<double,2> refsol({Nx,Nv});

  for(int i = 0; i<Nx; i++){
    for(int j = 0; j<Nv; j++){
      refsol(i,j) = cos(x[i] - tstar*v[j])*exp(-pow(v[j],2)/2);
    }
  }

  multi_array<double,2> sol_lr_tmp({r,Nv});
  multi_array<double,2> sol_lr({Nx,Nv});

  matmul_transb(lr_sol.S,lr_sol.V,sol_lr_tmp);
  matmul(lr_sol.X,sol_lr_tmp,sol_lr);

  double err = 0.0;
  double err_ptw = 0.0;

  for(int i = 0; i<Nx; i++){
    for(int j = 0; j<Nv; j++){
      err_ptw = abs(refsol(i,j) - sol_lr(i,j));
      if (err_ptw > err){
        err = err_ptw;
      }
    }
  }

  cout << err << endl;

  return 0;
}
