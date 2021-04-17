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

void save_vector(const multi_array<double,1>& u, std::string fn) {
    std::ofstream fs(fn.c_str(), std::ios::binary);
    fs.write((char*)u.data(), sizeof(double)*u.num_elements());
}


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

  int nsteps = tstar/tau;

  double ts_ee_forward = tau/nsteps_ee;
  double ts_ee_backward = -tau/nsteps_ee;

  double hx = (bx-ax) / Nx;
  double hv = (bv-av) / Nv;

  vector<double*> X, V;

  vector<double> x, v, x1, v1;

  for(Index i = 0; i < Nx; i++){
    x.push_back(ax + i*hx);
    x1.push_back(1.0 + alpha*cos(kappa*(ax + i*hx)));
  }


  for(Index i = 0; i < Nv; i++){
    v.push_back(av + i*hv);
    v1.push_back((1.0/sqrt(2*M_PI)) *exp(-pow((av + i*hv),2)/2.0));
  }

  X.push_back(x1.data());
  V.push_back(v1.data());

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

  std::function<double(double*,double*)> ip_x = inner_product_from_const_weight(hx, Nx);
  std::function<double(double*,double*)> ip_v = inner_product_from_const_weight(hv, Nv);

  lr2<double> lr0(r,{Nx,Nv});

  initialize(lr0, X, V, n_b, ip_x, ip_v);

  lr2<double> lr_sol(r,{Nx,Nv});

  // For FFT -- Pay attention we have to cast to int as Index seems not to work with fftw_many
  int rank = 1;
  int nx = int(Nx);
  int nv = int(Nv);
  int howmany = r;
  int istride = 1;
  int ostride = 1;
  int* inembedx = &nx;
  int* onembedx = &nx;
  int* inembedv = &nv;
  int* onembedv = &nv;
  Index idistx = Nx;
  Index odistx = Nx/2 + 1;
  Index idistv = Nv;
  Index odistv = Nv/2 + 1;

  multi_array<complex<double>,2> Khat({Nx/2+1,r});
  multi_array<complex<double>,2> Lhat({Nv/2+1,r});

  multi_array<complex<double>,2> Kehat({Nx/2+1,r});
  multi_array<complex<double>,2> Mhattmp({Nx/2+1,r});

  multi_array<complex<double>,2> Lvhat({Nv/2+1,r});
  multi_array<complex<double>,2> Nhattmp({Nv/2+1,r});


  fftw_plan px;
  px = fftw_plan_many_dft_r2c(rank, &nx, howmany, lr_sol.X.begin(), inembedx, istride, idistx, (fftw_complex*)Khat.begin(), onembedx, ostride, odistx, FFTW_MEASURE);

  fftw_plan qx;
  idistx = Nx/2 + 1;
  odistx = Nx;

  qx = fftw_plan_many_dft_c2r(rank, &nx, howmany, (fftw_complex*) Khat.begin(), inembedx, istride, idistx, lr_sol.X.begin(), onembedx, ostride, odistx, FFTW_MEASURE);

  fftw_plan pv;
  pv = fftw_plan_many_dft_r2c(rank, &nv, howmany, lr_sol.V.begin(), inembedv, istride, idistv, (fftw_complex*)Lhat.begin(), onembedv, ostride, odistv, FFTW_MEASURE);

  fftw_plan qv;
  idistv = Nv/2 + 1;
  odistv = Nv;

  qv = fftw_plan_many_dft_c2r(rank, &nv, howmany, (fftw_complex*) Lhat.begin(), inembedv, istride, idistv, lr_sol.V.begin(), onembedv, ostride, odistv, FFTW_MEASURE);

  // Mandatory to initialize after plan creation if we use FFTW_MEASURE
  lr_sol.X = lr0.X;
  lr_sol.S = lr0.S;
  lr_sol.V = lr0.V;

  vector<complex<double>> lambdax;
  for(int j = 0; j < (Nx/2 + 1) ; j++){
    lambdax.push_back(complex<double>(0.0,2.0*M_PI/(bx-ax)*j));
  }
  double ncx_backward = 1.0 / Nx;

  vector<complex<double>> lambdav;
  for(int j = 0; j < (Nv/2 + 1) ; j++){
    lambdav.push_back(complex<double>(0.0,2.0*M_PI/(bv-av)*j));
  }
  double ncv_backward = 1.0 / Nv;

  // For C coefficients
  multi_array<double,2> C1({r,r});
  multi_array<double,2> C2({r,r});
  multi_array<complex<double>,2> C2c({r,r});

  vector<double> wv;
  for(int j = 0; j < Nv; j++){
    wv.push_back(v[j] * hv);
  }

  multi_array<double,2> dV({Nv,r});


  // For Schur decomposition
  vector<double> dc_r,dc_i;
  dc_r.reserve(r);
  dc_i.reserve(r);

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
  dgees_(&jobvs,&sort,nullptr,&nn,T.begin(),&lda,&value,dc_r.data(),dc_i.data(),T.begin(),&ldvs,&work_opt,&lwork,nullptr,&info);

  lwork = int(work_opt);
  vector<double> work;
  work.reserve(lwork);

  // For D coefficients

  multi_array<double,2> D1({r,r});
  multi_array<double,2> D2({r,r});

  vector<double> wx;
  wx.reserve(Nx);

  multi_array<double,2> dX({Nx,r});

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({Nx});
  multi_array<complex<double>,1> efhat({Nx/2 + 1});

  fftw_plan pe;
  pe = fftw_plan_dft_r2c_1d(Nx, ef.begin(), (fftw_complex*) efhat.begin(), FFTW_MEASURE);

  fftw_plan qe;
  qe = fftw_plan_dft_c2r_1d(Nx, (fftw_complex*) efhat.begin(), ef.begin(), FFTW_MEASURE);

  multi_array<double,1> energy({nsteps});

  for(int i = 0; i < nsteps; i++){

    cout << "Time step " << i << " on " << nsteps << endl;

    /* K step */
    multi_array<double,2> tmpX(lr_sol.X);

    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field

    coeff_rho(lr_sol.V,hv,rho);

    rho *= -1.0;

    matvec(lr_sol.X,rho,ef);

    for(int ii = 0; ii < Nx; ii++){
        ef(ii) += 1.0;
    }

    fftw_execute_dft_r2c(pe,ef.begin(),(fftw_complex*)efhat.begin());

    efhat(0) = complex<double>(0.0,0.0);
    for(Index ii = 1; ii < (Nx/2+1); ii++){
      efhat(ii) /= lambdax[ii];
      efhat(ii) *= ncx_backward;
    }

    fftw_execute_dft_c2r(qe,(fftw_complex*)efhat.begin(),ef.begin());

    energy(i) = 0.0;
    for(int ii = 0; ii < Nx; ii++){
      energy(i) += 0.5*pow(ef(ii),2)*hx;
    }

    // Main of K step

    coeff(lr_sol.V, lr_sol.V, wv.data(), C1);

    fftw_execute_dft_r2c(pv,lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

    for(int k = 0; k<r; k++){
        for(int j = 0; j<(Nv/2 + 1); j++){
            Lhat(j,k) *= lambdav[j]*ncv_backward;
        }
    }

    fftw_execute_dft_c2r(qv,(fftw_complex*)Lhat.begin(),dV.begin());

    coeff(lr_sol.V, dV, hv, C2);


    multi_array<double,2> D_C(C1); // needed because dgees overwrites input, and I need C later on

    dgees_(&jobvs,&sort,nullptr,&nn,D_C.begin(),&lda,&value,dc_r.data(),dc_i.data(),T.begin(),&ldvs,work.data(),&lwork,nullptr,&info);

    // Forced casting as we need two complex matrices
    for(int j = 0; j < r; j++){
      for(int k = 0; k < r; k++){
        Tc(j,k) = complex<double>(T(j,k),0.0);
        C2c(j,k) = complex<double>(C2(j,k),0.0);
      }
    }

    fftw_execute_dft_r2c(px,lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    matmul(Khat,Tc,Mhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      for(int jj = 0; jj < r; jj++){
        for(int ii = 0; ii < Nx; ii++){
          lr_sol.X(ii,jj) *= ef(ii);
        }
      }

      fftw_execute_dft_r2c(px,lr_sol.X.begin(),(fftw_complex*)Kehat.begin());

      matmul_transb(Kehat,C2c,Khat);
      matmul(Khat,Tc,Mhattmp);

      for(int k = 0; k < r; k++){
        for(int j = 0; j < (Nx/2 + 1); j++){
          Mhat(j,k) *= exp(ts_ee_backward*lambdax[j]*dc_r[k]);
          Mhat(j,k) += ts_ee_forward*phi1(ts_ee_backward*lambdax[j]*dc_r[k])*Mhattmp(j,k);
        }
      }
      matmul_transb(Mhat,Tc,Khat);

      for(int k = 0; k < r; k++){
        for(int j = 0; j < (Nx/2 + 1); j++){
          Khat(j,k) *= ncx_backward;
        }
      }

      fftw_execute_dft_c2r(qx,(fftw_complex*)Khat.begin(),lr_sol.X.begin());

    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_x);

    /* S step */
    for(int ii = 0; ii < Nx; ii++){
      wx[ii] = hx*ef(ii);
    }

    coeff(lr_sol.X, lr_sol.X, wx.data(), D1);

    fftw_execute_dft_r2c(px,lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    for(int k = 0; k<r; k++){
        for(int j = 0; j<(Nx/2 + 1); j++){
            Khat(j,k) *= lambdax[j]*ncx_backward;
        }
    }

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

    multi_array<double,2> tmpV(lr_sol.V);
    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    D_C = D1; // needed because dgees overwrites input, and I need C later on

    dgees_(&jobvs,&sort,nullptr,&nn,D_C.begin(),&lda,&value,dc_r.data(),dc_i.data(),T.begin(),&ldvs,work.data(),&lwork,nullptr,&info);

    // Forced casting as we need two complex matrices
    for(int j = 0; j < r; j++){
      for(int k = 0; k < r; k++){
        Tc(j,k) = complex<double>(T(j,k),0.0);
        C2c(j,k) = complex<double>(D2(j,k),0.0);
      }
    }

    fftw_execute_dft_r2c(pv,lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

    matmul(Lhat,Tc,Nhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      for(int jj = 0; jj < r; jj++){
        for(int ii = 0; ii < Nv; ii++){
          lr_sol.V(ii,jj) *= v[ii];
        }
      }

      fftw_execute_dft_r2c(pv,lr_sol.V.begin(),(fftw_complex*)Lvhat.begin());

      matmul_transb(Lvhat,C2c,Lhat);
      matmul(Lhat,Tc,Nhattmp);

      for(int k = 0; k < r; k++){
        for(int j = 0; j < (Nv/2 + 1); j++){
          Nhat(j,k) *= exp(ts_ee_forward*lambdav[j]*dc_r[k]);
          Nhat(j,k) += ts_ee_backward*phi1(ts_ee_forward*lambdav[j]*dc_r[k])*Nhattmp(j,k);
        }
      }
      matmul_transb(Nhat,Tc,Lhat);

      for(int k = 0; k < r; k++){
        for(int j = 0; j < (Nv/2 + 1); j++){
          Lhat(j,k) *= ncv_backward;
        }
      }

      fftw_execute_dft_c2r(qv,(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_v);

    transpose_inplace(lr_sol.S);

    cout.precision(15);
    cout << energy(i) << endl;

  }

  save_vector(energy,"energy.bin");

  return 0;
}
