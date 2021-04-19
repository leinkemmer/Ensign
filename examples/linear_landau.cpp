#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>

#include <random>
#include <complex>
#include <fftw3.h>
#include <cstring>

int main(){

  Index Nx = 64; // NEEDS TO BE EVEN FOR FOURIER
  Index Nv = 256; // NEEDS TO BE EVEN FOR FOURIER

  int r = 5; // rank desired
  int n_b = 1; // number of actual basis functions

  double tstar = 100; // final time
  double tau = 0.0125; // time step splitting

  int nsteps_ee = 50; // number of time steps for explicit euler

  double ax = 0;
  double bx = 4.0*M_PI;

  double av = -6.0;
  double bv = 6.0;

  double alpha = 0.01;
  double kappa = 0.5;

  Index nsteps = tstar/tau;

  double ts_ee_forward = tau/nsteps_ee;
  double ts_ee_backward = -tau/nsteps_ee;

  double hx = (bx-ax) / Nx;
  double hv = (bv-av) / Nv;

  vector<double*> X, V; // ONLY VECTOR SURVIVED

  multi_array<double,1> x({Nx});
  multi_array<double,1> x1({Nx});

  for(Index i = 0; i < Nx; i++){
    x(i) = ax + i*hx;
    x1(i) = 1.0 + alpha*cos(kappa*(ax + i*hx));
  }

  multi_array<double,1> v({Nv});
  multi_array<double,1> v1({Nv});

  for(Index i = 0; i < Nv; i++){
    v(i) = av + i*hv;
    v1(i) = (1.0/sqrt(2*M_PI)) *exp(-pow((av + i*hv),2)/2.0);
  }

  X.push_back(x1.begin());
  V.push_back(v1.begin());

  multi_array<double,2> RX({Nx,r-n_b});
  multi_array<double,2> RV({Nv,r-n_b});

  std::default_random_engine generator(time(0));
  std::normal_distribution<double> distribution(0.0,1.0);

  for(Index j = 0; j < (r-n_b); j++){
    for(Index i = 0; i < Nx; i++){
      RX(i,j) = distribution(generator);
    }
    for(Index i = 0; i < Nv; i++){
      RV(i,j) = distribution(generator);
    }

    X.push_back(RX.begin() + j*Nx);
    V.push_back(RV.begin() + j*Nv);
  }

  std::function<double(double*,double*)> ip_x = inner_product_from_const_weight(hx, Nx);
  std::function<double(double*,double*)> ip_v = inner_product_from_const_weight(hv, Nv);

  lr2<double> lr0(r,{Nx,Nv});

  initialize(lr0, X, V, n_b, ip_x, ip_v);

  lr2<double> lr_sol(r,{Nx,Nv});

  // For FFT -- Pay attention we have to cast to int as Index seems not to work with fftw_many
  int nx = int(Nx);
  int nv = int(Nv);

  multi_array<complex<double>,2> Khat({Nx/2+1,r});
  multi_array<complex<double>,2> Lhat({Nv/2+1,r});

  fftw_plan px = fftw_plan_many_dft_r2c(1, &nx, r, lr_sol.X.begin(), &nx, 1, Nx, (fftw_complex*)Khat.begin(), &nx, 1, Nx/2 + 1, FFTW_MEASURE);
  fftw_plan qx = fftw_plan_many_dft_c2r(1, &nx, r, (fftw_complex*) Khat.begin(), &nx, 1, Nx/2 + 1, lr_sol.X.begin(), &nx, 1, Nx, FFTW_MEASURE);

  fftw_plan pv = fftw_plan_many_dft_r2c(1, &nv, r, lr_sol.V.begin(), &nv, 1, Nv, (fftw_complex*)Lhat.begin(), &nv, 1, Nv/2 + 1, FFTW_MEASURE);
  fftw_plan qv = fftw_plan_many_dft_c2r(1, &nv, r, (fftw_complex*) Lhat.begin(), &nv, 1, Nv/2 + 1, lr_sol.V.begin(), &nv, 1, Nv, FFTW_MEASURE);

  // Mandatory to initialize after plan creation if we use FFTW_MEASURE
  lr_sol.X = lr0.X;
  lr_sol.S = lr0.S;
  lr_sol.V = lr0.V;

  multi_array<complex<double>,2> Kehat({Nx/2+1,r});
  multi_array<complex<double>,2> Mhattmp({Nx/2+1,r});

  multi_array<complex<double>,2> Lvhat({Nv/2+1,r});
  multi_array<complex<double>,2> Nhattmp({Nv/2+1,r});

  multi_array<complex<double>,1> lambdax({Nx/2+1});
  multi_array<complex<double>,1> lambdax_n({Nx/2+1});

  double ncx = 1.0 / Nx;

  for(Index j = 0; j < (Nx/2 + 1) ; j++){
    lambdax(j) = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);
    lambdax_n(j) = complex<double>(0.0,2.0*M_PI/(bx-ax)*j*ncx);
  }

  multi_array<complex<double>,1> lambdav({Nv/2+1});
  multi_array<complex<double>,1> lambdav_n({Nv/2+1});

  double ncv = 1.0 / Nv;

  for(Index j = 0; j < (Nv/2 + 1) ; j++){
    lambdav(j) = complex<double>(0.0,2.0*M_PI/(bv-av)*j);
    lambdav_n(j) = complex<double>(0.0,2.0*M_PI/(bv-av)*j*ncv);
  }

  // For C coefficients
  multi_array<double,2> C1({r,r});
  multi_array<double,2> C2({r,r});
  multi_array<complex<double>,2> C2c({r,r});

  multi_array<double,1> wv({Nv});

  for(Index j = 0; j < Nv; j++){
    wv(j) = v(j) * hv;
  }

  multi_array<double,2> dV({Nv,r});

  // For Schur decomposition
  multi_array<double,1> dc_r({r});
  multi_array<double,1> dc_i({r});

  multi_array<double,2> T({r,r});
  multi_array<double,2> multmp({r,r});

  multi_array<complex<double>,2> Mhat({Nx/2 + 1,r});
  multi_array<complex<double>,2> Nhat({Nv/2 + 1,r});
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
  dgees_(&jobvs,&sort,nullptr,&nn,T.begin(),&lda,&value,dc_r.begin(),dc_i.begin(),T.begin(),&ldvs,&work_opt,&lwork,nullptr,&info);

  lwork = int(work_opt);
  multi_array<double,1> work({lwork});

  // For D coefficients

  multi_array<double,2> D1({r,r});
  multi_array<double,2> D2({r,r});

  multi_array<double,1> wx({Nx});
  multi_array<double,2> dX({Nx,r});

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({Nx});
  multi_array<complex<double>,1> efhat({Nx/2 + 1});

  fftw_plan pe = fftw_plan_dft_r2c_1d(Nx, ef.begin(), (fftw_complex*) efhat.begin(), FFTW_MEASURE);
  fftw_plan qe = fftw_plan_dft_c2r_1d(Nx, (fftw_complex*) efhat.begin(), ef.begin(), FFTW_MEASURE);

  multi_array<double,1> energy({nsteps});

  multi_array<double,2> tmpX({Nx,r});
  multi_array<double,2> tmpV({Nv,r});

  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i << " on " << nsteps << endl;

    /* K step */
    tmpX = lr_sol.X;

    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field

    coeff_rho(lr_sol.V,hv,rho);

    rho *= -1.0;

    matvec(lr_sol.X,rho,ef);

    for(Index ii = 0; ii < Nx; ii++){
        ef(ii) += 1.0;
    }

    fftw_execute_dft_r2c(pe,ef.begin(),(fftw_complex*)efhat.begin());

    efhat(0) = complex<double>(0.0,0.0);
    for(Index ii = 1; ii < (Nx/2+1); ii++){
      efhat(ii) /= (lambdax(ii)/ncx);
    }

    fftw_execute_dft_c2r(qe,(fftw_complex*)efhat.begin(),ef.begin());

    energy(i) = 0.0;
    for(Index ii = 0; ii < Nx; ii++){
      energy(i) += 0.5*pow(ef(ii),2)*hx;
    }

    // Main of K step

    coeff(lr_sol.V, lr_sol.V, wv.begin(), C1);

    fftw_execute_dft_r2c(pv,lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

    ptw_mult_row(Lhat,lambdav_n.begin(),Lhat);

    fftw_execute_dft_c2r(qv,(fftw_complex*)Lhat.begin(),dV.begin());

    coeff(lr_sol.V, dV, hv, C2);

    multi_array<double,2> D_C(C1); // needed because dgees overwrites input, and I need C later on

    dgees_(&jobvs,&sort,nullptr,&nn,D_C.begin(),&lda,&value,dc_r.begin(),dc_i.begin(),T.begin(),&ldvs,work.begin(),&lwork,nullptr,&info);

    T.to_cplx(Tc);
    C2.to_cplx(C2c);

    fftw_execute_dft_r2c(px,lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    matmul(Khat,Tc,Mhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row(lr_sol.X,ef.begin(),lr_sol.X);

      fftw_execute_dft_r2c(px,lr_sol.X.begin(),(fftw_complex*)Kehat.begin());

      matmul_transb(Kehat,C2c,Khat);
      matmul(Khat,Tc,Mhattmp);

      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          Mhat(j,k) *= exp(ts_ee_backward*lambdax(j)*dc_r(k));
          Mhat(j,k) += ts_ee_forward*phi1(ts_ee_backward*lambdax(j)*dc_r(k))*Mhattmp(j,k);
        }
      }
      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(qx,(fftw_complex*)Khat.begin(),lr_sol.X.begin());

    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_x);

    /* S step */
    for(Index ii = 0; ii < Nx; ii++){
      wx(ii) = hx*ef(ii);
    }

    coeff(lr_sol.X, lr_sol.X, wx.begin(), D1);

    fftw_execute_dft_r2c(px,lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    ptw_mult_row(Khat,lambdax_n.begin(),Khat);

    fftw_execute_dft_c2r(qx,(fftw_complex*)Khat.begin(),dX.begin());

    coeff(lr_sol.X, dX, hx, D2);

    // Explicit Euler
    for(int j = 0; j< nsteps_ee; j++){
      matmul_transb(lr_sol.S,C1,D_C);
      matmul(D2,D_C,T);

      matmul_transb(lr_sol.S,C2,D_C);
      matmul(D1,D_C,multmp);

      T -= multmp;
      T *= ts_ee_forward;
      lr_sol.S += T;
    }

    /* L step */

    tmpV = lr_sol.V;
    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    D_C = D1; // needed because dgees overwrites input, and I need C later on

    dgees_(&jobvs,&sort,nullptr,&nn,D_C.begin(),&lda,&value,dc_r.begin(),dc_i.begin(),T.begin(),&ldvs,work.begin(),&lwork,nullptr,&info);

    T.to_cplx(Tc);
    D2.to_cplx(C2c);

    fftw_execute_dft_r2c(pv,lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

    matmul(Lhat,Tc,Nhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row(lr_sol.V,v.begin(),lr_sol.V);

      fftw_execute_dft_r2c(pv,lr_sol.V.begin(),(fftw_complex*)Lvhat.begin());

      matmul_transb(Lvhat,C2c,Lhat);
      matmul(Lhat,Tc,Nhattmp);

      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nv/2 + 1); j++){
          Nhat(j,k) *= exp(ts_ee_forward*lambdav(j)*dc_r(k));
          Nhat(j,k) += ts_ee_backward*phi1(ts_ee_forward*lambdav(j)*dc_r(k))*Nhattmp(j,k);
        }
      }
      matmul_transb(Nhat,Tc,Lhat);

      Lhat *= ncv;
      fftw_execute_dft_c2r(qv,(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_v);

    transpose_inplace(lr_sol.S);

    cout.precision(15);
    cout << energy(i) << endl;

  }

  energy.save_vector("energy.bin");

  return 0;
}
