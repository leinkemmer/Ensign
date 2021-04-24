#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>

#include <random>
#include <complex>
#include <fftw3.h>
#include <cstring>

int main(){

  Index Nx = 32; // NEEDS TO BE EVEN FOR FOURIER
  Index Ny = 32; // NEEDS TO BE EVEN FOR FOURIER

  Index Nv = 64; // NEEDS TO BE EVEN FOR FOURIER
  Index Nw = 64; // NEEDS TO BE EVEN FOR FOURIER

  int r = 5; // rank desired
  int n_b = 1; // number of actual basis functions

  double tstar = 100; // final time
  double tau = 0.025; // time step splitting

  double ax = 0;
  double bx = 4.0*M_PI;
  double ay = 0;
  double by = 4.0*M_PI;

  double av = -6.0;
  double bv = 6.0;
  double aw = -6.0;
  double bw = 6.0;

  double alpha = 0.01;
  double kappa1 = 0.5;
  double kappa2 = 0.5;

  Index nsteps_split = 10; // Number of time steps internal splitting
  Index nsteps_ee = 10; // Number of time steps of exponential euler in internal splitting

  Index nsteps = tstar/tau;

  //nsteps = 1;

  double ts_split = tau / nsteps_split;
  double ts_ee = ts_split / nsteps_ee;

  double hx = (bx-ax) / Nx;
  double hy = (by-ay) / Ny;

  double hv = (bv-av) / Nv;
  double hw = (bw-aw) / Nw;

  vector<double*> X, V; // ONLY VECTOR SURVIVED

  multi_array<double,1> x({Nx*Ny});
  multi_array<double,1> y({Nx*Ny});

  multi_array<double,1> xx({Nx*Ny});

  for(Index j = 0; j < Ny; j++){
    for(Index i = 0; i < Nx; i++){
        x(i+j*Nx) = ax + i*hx;
        y(i+j*Nx) = ay + j*hy;

        xx(i+j*Nx) = 1.0 + alpha*cos(kappa1*x(i+j*Nx)) + alpha*cos(kappa2*y(i+j*Nx));
    }
  }

  multi_array<double,1> v({Nv*Nw});
  multi_array<double,1> w({Nv*Nw});

  multi_array<double,1> vv({Nv*Nw});

  for(Index j = 0; j < Nw; j++){
    for(Index i = 0; i < Nv; i++){
      v(i+j*Nv) = av + i*hv;
      w(i+j*Nv) = aw + j*hw;

      vv(i+j*Nv) = (1.0/(2*M_PI)) *exp(-(pow(v(i+j*Nv),2)+pow(w(i+j*Nv),2))/2.0);
    }
  }

  X.push_back(xx.begin());
  V.push_back(vv.begin());

  multi_array<double,2> RX({Nx*Ny,r-n_b});
  multi_array<double,2> RV({Nv*Nw,r-n_b});

  std::default_random_engine generator(time(0));
  std::normal_distribution<double> distribution(0.0,1.0);

  for(Index j = 0; j < (r-n_b); j++){
    for(Index i = 0; i < (Nx*Ny); i++){
      RX(i,j) = distribution(generator);
      //RX(i,j) = i + j;
    }
    for(Index i = 0; i < (Nv*Nw); i++){
      RV(i,j) = distribution(generator);
      //RV(i,j) = i + j;
    }

    X.push_back(RX.begin() + j*(Nx*Ny));
    V.push_back(RV.begin() + j*(Nv*Nw));
  }

  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(hx*hy, Nx*Ny);
  std::function<double(double*,double*)> ip_vv = inner_product_from_const_weight(hv*hw, Nv*Nw);

  lr2<double> lr0(r,{Nx*Ny,Nv*Nw});

  initialize(lr0, X, V, n_b, ip_xx, ip_vv);

  lr2<double> lr_sol(r,{Nx*Ny,Nv*Nw});

  multi_array<complex<double>,1> lambdax({Ny*(Nx/2+1)});
  multi_array<complex<double>,1> lambday({Ny*(Nx/2+1)});
  multi_array<complex<double>,1> lambdax_n({Ny*(Nx/2+1)});
  multi_array<complex<double>,1> lambday_n({Ny*(Nx/2+1)});

  double ncxx = 1.0 / (Nx*Ny);

  for(Index j = 0; j < Ny; j++){
    for(Index i = 0; i < (Nx/2+1); i++){
        lambdax(i+j*(Nx/2+1)) = complex<double>(0.0,2.0*M_PI/(bx-ax)*i);
        lambdax_n(i+j*(Nx/2+1)) = complex<double>(0.0,2.0*M_PI/(bx-ax)*i)*ncxx;
      }
  }

  // Think about a smarter way to define them
  for(Index j = 0; j < (Ny/2+1); j++){
    for(Index i = 0; i < (Nx/2+1); i++){
        lambday(i+j*(Nx/2+1)) = complex<double>(0.0,2.0*M_PI/(by-ay)*j);
        lambday_n(i+j*(Nx/2+1)) = complex<double>(0.0,2.0*M_PI/(by-ay)*j)*ncxx;
      }
  }

  for(Index j = (Ny/2+1); j < Ny; j++){
    for(Index i = 0; i < (Nx/2+1); i++){
        lambday(i+j*(Nx/2+1)) = complex<double>(0.0,2.0*M_PI/(by-ay)*(j-Ny));
        lambday_n(i+j*(Nx/2+1)) = complex<double>(0.0,2.0*M_PI/(by-ay)*(j-Ny))*ncxx;
      }
  }

  multi_array<complex<double>,1> lambdav({Nw*(Nv/2+1)});
  multi_array<complex<double>,1> lambdaw({Nw*(Nv/2+1)});
  multi_array<complex<double>,1> lambdav_n({Nw*(Nv/2+1)});
  multi_array<complex<double>,1> lambdaw_n({Nw*(Nv/2+1)});

  double ncvv = 1.0 / (Nv*Nw);

  for(Index j = 0; j < Nw; j++){
    for(Index i = 0; i < (Nv/2+1); i++){
        lambdav(i+j*(Nv/2+1)) = complex<double>(0.0,2.0*M_PI/(bv-av)*i);
        lambdav_n(i+j*(Nv/2+1)) = complex<double>(0.0,2.0*M_PI/(bv-av)*i)*ncvv;
      }
  }

  // Think about a smarter way to define them
  for(Index j = 0; j < (Nw/2+1); j++){
    for(Index i = 0; i < (Nv/2+1); i++){
        lambdaw(i+j*(Nv/2+1)) = complex<double>(0.0,2.0*M_PI/(bw-aw)*j);
        lambdaw_n(i+j*(Nv/2+1)) = complex<double>(0.0,2.0*M_PI/(bw-aw)*j)*ncvv;
      }
  }

  for(Index j = (Nw/2+1); j < Nw; j++){
    for(Index i = 0; i < (Nv/2+1); i++){
        lambdaw(i+j*(Nv/2+1)) = complex<double>(0.0,2.0*M_PI/(bw-aw)*(j-Nw));
        lambdaw_n(i+j*(Nv/2+1)) = complex<double>(0.0,2.0*M_PI/(bw-aw)*(j-Nw))*ncvv;
      }
  }


  // For FFT -- Pay attention we have to cast to int as Index seems not to work with fftw_many
  //int nx = int(Nx);
  //int nv = int(Nv);

  multi_array<int,1> dimsx({2});
  dimsx(0) = int(Ny);
  dimsx(1) = int(Nx);
  // ocio che sono swappate perche facciamo per colmajor

  multi_array<complex<double>,2> Khat({Ny*(Nx/2+1),r});
  multi_array<complex<double>,2> Khattmp({Ny*(Nx/2+1),r});

  fftw_plan px = fftw_plan_many_dft_r2c(2, dimsx.begin(), r, lr_sol.X.begin(), NULL, 1, Nx*Ny, (fftw_complex*)Khat.begin(), NULL, 1, Ny*(Nx/2 + 1), FFTW_MEASURE);
  fftw_plan qx = fftw_plan_many_dft_c2r(2, dimsx.begin(), r, (fftw_complex*) Khat.begin(), NULL, 1, Ny*(Nx/2 + 1), lr_sol.X.begin(), NULL, 1, Nx*Ny, FFTW_MEASURE);

  multi_array<int,1> dimsv({2});
  dimsv(0) = int(Nw);
  dimsv(1) = int(Nv);
  // ocio che sono swappate perche facciamo per colmajor

  multi_array<complex<double>,2> Lhat({Nw*(Nv/2+1),r});
  multi_array<complex<double>,2> Lhattmp({Nw*(Nv/2+1),r});

  fftw_plan pv = fftw_plan_many_dft_r2c(2, dimsv.begin(), r, lr_sol.V.begin(), NULL, 1, Nv*Nw, (fftw_complex*)Lhat.begin(), NULL, 1, Nw*(Nv/2 + 1), FFTW_MEASURE);
  fftw_plan qv = fftw_plan_many_dft_c2r(2, dimsv.begin(), r, (fftw_complex*) Lhat.begin(), NULL, 1, Nw*(Nv/2 + 1), lr_sol.V.begin(), NULL, 1, Nv*Nw, FFTW_MEASURE);

  // Mandatory to initialize after plan creation if we use FFTW_MEASURE
  lr_sol.X = lr0.X;
  lr_sol.S = lr0.S;
  lr_sol.V = lr0.V;

  multi_array<double,2> Kex({Nx*Ny,r});
  multi_array<double,2> Key({Nx*Ny,r});

  multi_array<complex<double>,2> Kexhat({Ny*(Nx/2+1),r});
  multi_array<complex<double>,2> Keyhat({Ny*(Nx/2+1),r});

  multi_array<complex<double>,2> Mhattmp({Ny*(Nx/2+1),r});

  multi_array<double,2> Lv({Nv*Nw,r});
  multi_array<double,2> Lw({Nv*Nw,r});

  multi_array<complex<double>,2> Lvhat({Nw*(Nv/2+1),r});
  multi_array<complex<double>,2> Lwhat({Nw*(Nv/2+1),r});

  multi_array<complex<double>,2> Nhattmp({Nw*(Nv/2+1),r});

  // For C coefficients
  multi_array<double,2> C1v({r,r});
  multi_array<double,2> C1w({r,r});

  multi_array<double,2> C2v({r,r});
  multi_array<double,2> C2w({r,r});

  multi_array<complex<double>,2> C2vc({r,r});
  multi_array<complex<double>,2> C2wc({r,r});

  multi_array<double,1> we_v({Nv*Nw});
  multi_array<double,1> we_w({Nv*Nw});

  for(Index j = 0; j < (Nv*Nw); j++){
    we_v(j) = v(j) * hv * hw;
    we_w(j) = w(j) * hv * hw;
  }

  multi_array<double,2> dV_v({Nv*Nw,r});
  multi_array<double,2> dV_w({Nv*Nw,r});

  multi_array<complex<double>,2> dVhat_v({Nw*(Nv/2+1),r});
  multi_array<complex<double>,2> dVhat_w({Nw*(Nv/2+1),r});

  // For Schur decomposition
  multi_array<double,1> dcv_r({r});
  multi_array<double,1> dcv_i({r});
  multi_array<double,1> dcw_r({r});
  multi_array<double,1> dcw_i({r});

  multi_array<double,2> Tv({r,r});
  multi_array<double,2> Tw({r,r});

  //multi_array<double,2> multmp({r,r});

  multi_array<complex<double>,2> Mhat({Ny*(Nx/2 + 1),r});
  multi_array<complex<double>,2> Nhat({Nw*(Nv/2 + 1),r});
  multi_array<complex<double>,2> Tvc({r,r});
  multi_array<complex<double>,2> Twc({r,r});

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
  dgees_(&jobvs,&sort,nullptr,&nn,Tv.begin(),&lda,&value,dcv_r.begin(),dcv_i.begin(),Tv.begin(),&ldvs,&work_opt,&lwork,nullptr,&info);

  lwork = int(work_opt);
  multi_array<double,1> work({lwork});

  // For D coefficients

  multi_array<double,2> D1x({r,r});
  multi_array<double,2> D1y({r,r});

  multi_array<double,2> D2x({r,r});
  multi_array<double,2> D2y({r,r});

  multi_array<complex<double>,2> D2xc({r,r});
  multi_array<complex<double>,2> D2yc({r,r});

  multi_array<double,1> we_x({Nx*Ny});
  multi_array<double,1> we_y({Nx*Ny});

  multi_array<double,2> dX_x({Nx*Ny,r});
  multi_array<double,2> dX_y({Nx*Ny,r});

  multi_array<complex<double>,2> dXhat_x({Ny*(Nx/2+1),r});
  multi_array<complex<double>,2> dXhat_y({Ny*(Nx/2+1),r});

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({Nx*Ny});
  multi_array<double,1> efx({Nx*Ny});
  multi_array<double,1> efy({Nx*Ny});

  multi_array<complex<double>,1> efhat({Ny*(Nx/2 + 1)});
  multi_array<complex<double>,1> efhatx({Ny*(Nx/2 + 1)});
  multi_array<complex<double>,1> efhaty({Ny*(Nx/2 + 1)});

  fftw_plan pe = fftw_plan_dft_r2c_2d(Ny,Nx, ef.begin(), (fftw_complex*) efhat.begin(), FFTW_MEASURE);
  fftw_plan qe = fftw_plan_dft_c2r_2d(Ny, Nx, (fftw_complex*) efhat.begin(), ef.begin(), FFTW_MEASURE);

  multi_array<double,1> energy({nsteps});

  multi_array<double,2> tmpX({Nx*Ny,r});
  multi_array<double,2> tmpV({Nv*Nw,r});

//  cout << lr_sol.V << endl;


  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i << " on " << nsteps << endl;

    tmpX = lr_sol.X;

    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field

    coeff_rho(lr_sol.V,hv*hw,rho);

    rho *= -1.0;

    matvec(lr_sol.X,rho,ef);

    for(Index ii = 0; ii < Nx*Ny; ii++){
        ef(ii) += 1.0;
    }

    fftw_execute_dft_r2c(pe,ef.begin(),(fftw_complex*)efhat.begin());

    for(Index j = 0; j < Ny; j++){
      for(Index i = 0; i < (Nx/2+1); i++){
        efhatx(i+j*(Nx/2+1)) = efhat(i+j*(Nx/2+1)) * lambdax(i+j*(Nx/2+1)) / (pow(lambdax(i+j*(Nx/2+1)),2) + pow(lambday(i+j*(Nx/2+1)),2)) * ncxx;
        efhaty(i+j*(Nx/2+1)) = efhat(i+j*(Nx/2+1)) * lambday(i+j*(Nx/2+1)) / (pow(lambdax(i+j*(Nx/2+1)),2) + pow(lambday(i+j*(Nx/2+1)),2)) * ncxx ;
      }
    }
    efhatx(0) = complex<double>(0.0,0.0);
    efhaty(0) = complex<double>(0.0,0.0);

    fftw_execute_dft_c2r(qe,(fftw_complex*)efhatx.begin(),efx.begin());
    fftw_execute_dft_c2r(qe,(fftw_complex*)efhaty.begin(),efy.begin());

    energy(i) = 0.0;
    for(Index ii = 0; ii < (Nx*Ny); ii++){
      energy(i) += 0.5*(pow(efx(ii),2)+pow(efy(ii),2))*hx*hy;
    }

    // Main of K step

    coeff(lr_sol.V, lr_sol.V, we_v.begin(), C1v);
    coeff(lr_sol.V, lr_sol.V, we_w.begin(), C1w);

    fftw_execute_dft_r2c(pv,lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

    ptw_mult_row(Lhat,lambdav_n.begin(),dVhat_v);
    ptw_mult_row(Lhat,lambdaw_n.begin(),dVhat_w);

    fftw_execute_dft_c2r(qv,(fftw_complex*)dVhat_v.begin(),dV_v.begin());
    fftw_execute_dft_c2r(qv,(fftw_complex*)dVhat_w.begin(),dV_w.begin());

    coeff(lr_sol.V, dV_v, hv*hw, C2v);
    coeff(lr_sol.V, dV_w, hv*hw, C2w);

    cout.precision(15);
    cout << std::scientific;

    multi_array<double,2> D_C1v(C1v); // needed because dgees overwrites input, and I need C later on
    multi_array<double,2> D_C1w(C1w); // needed because dgees overwrites input, and I need C later on

    dgees_(&jobvs,&sort,nullptr,&nn,D_C1v.begin(),&lda,&value,dcv_r.begin(),dcv_i.begin(),Tv.begin(),&ldvs,work.begin(),&lwork,nullptr,&info);
    dgees_(&jobvs,&sort,nullptr,&nn,D_C1w.begin(),&lda,&value,dcw_r.begin(),dcw_i.begin(),Tw.begin(),&ldvs,work.begin(),&lwork,nullptr,&info);

    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    C2v.to_cplx(C2vc);
    C2w.to_cplx(C2wc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step
      fftw_execute_dft_r2c(px,lr_sol.X.begin(),(fftw_complex*)Khat.begin());
      matmul(Khat,Tvc,Mhat);

      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Ny*(Nx/2 + 1)); j++){
          Mhat(j,k) *= exp(-ts_split*lambdax(j)*dcv_r(k));
        }
      }

      matmul_transb(Mhat,Tvc,Khat);
      Khat *= ncxx;
      fftw_execute_dft_c2r(qx,(fftw_complex*)Khat.begin(),lr_sol.X.begin());

      // Full step
      fftw_execute_dft_r2c(px,lr_sol.X.begin(),(fftw_complex*)Khat.begin()); // Posso evitarlo evitando la moltiplic. ncxx e usando Khat
      matmul(Khat,Twc,Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.X,efx.begin(),Kex);
        ptw_mult_row(lr_sol.X,efy.begin(),Key);

        fftw_execute_dft_r2c(px,Kex.begin(),(fftw_complex*)Kexhat.begin());
        fftw_execute_dft_r2c(px,Key.begin(),(fftw_complex*)Keyhat.begin());

        matmul_transb(Kexhat,C2vc,Khat);
        matmul_transb(Keyhat,C2wc,Khattmp);

        Khat += Khattmp;

        matmul(Khat,Twc,Mhattmp);

        for(int k = 0; k < r; k++){
          for(Index j = 0; j < (Ny*(Nx/2 + 1)); j++){
            Mhat(j,k) *= exp(-ts_ee*lambday(j)*dcw_r(k));
            Mhat(j,k) += ts_ee*phi1(-ts_ee*lambday(j)*dcw_r(k))*Mhattmp(j,k);
          }
        }

        matmul_transb(Mhat,Twc,Khat);

        Khat *= ncxx;
        fftw_execute_dft_c2r(qx,(fftw_complex*)Khat.begin(),lr_sol.X.begin());
      }
    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_xx);

    // S Step

    for(Index j = 0; j < (Nx*Ny); j++){
      we_x(j) = efx(j) * hx * hy;
      we_y(j) = efy(j) * hx * hy;
    }

    coeff(lr_sol.X, lr_sol.X, we_x.begin(), D1x);
    coeff(lr_sol.X, lr_sol.X, we_y.begin(), D1y);

    fftw_execute_dft_r2c(px,lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    ptw_mult_row(Khat,lambdax_n.begin(),dXhat_x);
    ptw_mult_row(Khat,lambday_n.begin(),dXhat_y);

    fftw_execute_dft_c2r(qx,(fftw_complex*)dXhat_x.begin(),dX_x.begin());
    fftw_execute_dft_c2r(qx,(fftw_complex*)dXhat_y.begin(),dX_y.begin());

    coeff(lr_sol.X, dX_x, hx*hy, D2x);
    coeff(lr_sol.X, dX_y, hx*hy, D2y);


    for(Index jj = 0; jj< nsteps_split; jj++){
      matmul_transb(lr_sol.S,C1v,D_C1v);
      matmul(D2x,D_C1v,Tv);

      matmul_transb(lr_sol.S,C1w,D_C1v);
      matmul(D2y,D_C1v,Tw);

      Tv += Tw;

      matmul_transb(lr_sol.S,C2v,D_C1v);
      matmul(D1x,D_C1v,Tw);

      Tv -= Tw;

      matmul_transb(lr_sol.S,C2w,D_C1v);
      matmul(D1y,D_C1v,Tw);

      Tv -= Tw;

      Tv *= ts_split;
      lr_sol.S += Tv;
    }

    // L step

    tmpV = lr_sol.V;

    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    D_C1v = D1x; // needed because dgees overwrites input, and I need C later on
    D_C1w = D1y; // needed because dgees overwrites input, and I need C later on

    dgees_(&jobvs,&sort,nullptr,&nn,D_C1v.begin(),&lda,&value,dcv_r.begin(),dcv_i.begin(),Tv.begin(),&ldvs,work.begin(),&lwork,nullptr,&info);
    dgees_(&jobvs,&sort,nullptr,&nn,D_C1w.begin(),&lda,&value,dcw_r.begin(),dcw_i.begin(),Tw.begin(),&ldvs,work.begin(),&lwork,nullptr,&info);

    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    D2x.to_cplx(C2vc);
    D2y.to_cplx(C2wc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
    //  cout << ii << endl;
      // Full step
      fftw_execute_dft_r2c(pv,lr_sol.V.begin(),(fftw_complex*)Lhat.begin());
      matmul(Lhat,Tvc,Nhat);

      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nw*(Nv/2 + 1)); j++){
          Nhat(j,k) *= exp(ts_split*lambdav(j)*dcv_r(k));
        }
      }

      matmul_transb(Nhat,Tvc,Lhat);
      Lhat *= ncvv;
      fftw_execute_dft_c2r(qv,(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      // Full step
      fftw_execute_dft_r2c(pv,lr_sol.V.begin(),(fftw_complex*)Lhat.begin()); // Posso evitarlo evitando la moltiplic. ncxx e usando Khat
      matmul(Lhat,Twc,Nhat);

//      cout << Nhat << endl;

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.V,v.begin(),Lv);
        ptw_mult_row(lr_sol.V,w.begin(),Lw);

        fftw_execute_dft_r2c(pv,Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(pv,Lw.begin(),(fftw_complex*)Lwhat.begin());

        matmul_transb(Lvhat,C2vc,Lhat);
        matmul_transb(Lwhat,C2wc,Lhattmp);

        Lhat += Lhattmp;

        matmul(Lhat,Twc,Nhattmp);

        for(int k = 0; k < r; k++){
          for(Index j = 0; j < (Nw*(Nv/2 + 1)); j++){
            Nhat(j,k) *= exp(ts_ee*lambdaw(j)*dcw_r(k));
            Nhat(j,k) -= ts_ee*phi1(ts_ee*lambdaw(j)*dcw_r(k))*Nhattmp(j,k);
          }
        }

        matmul_transb(Nhat,Twc,Lhat);

        Lhat *= ncvv;
        fftw_execute_dft_c2r(qv,(fftw_complex*)Lhat.begin(),lr_sol.V.begin());
      }

    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_vv);

    transpose_inplace(lr_sol.S);

    cout << energy(i) << endl;

  }


  energy.save_vector("energy_2d.bin");

  return 0;
}
