#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>

#include <random>
#include <complex>
#include <cstring>

int main(){

  ofstream time_error_1d;
  ofstream tau_span_1d;
  time_error_1d.open("time_error_1d.txt");
  tau_span_1d.open("tau_span_1d.txt");

  Index Nx = 64; // NEEDS TO BE EVEN FOR FOURIER
  Index Nv = 256; // NEEDS TO BE EVEN FOR FOURIER

  int r = 3; // rank desired

  double tstar = 10; // final time
  double tauref = pow(2,-10);
  double tau = tauref; // time step splitting
  int i_err = 0;

  int nsteps_ee = 2; // number of time steps for explicit euler

  double ax = 0;
  double bx = 4.0*M_PI;

  double av = -6.0;
  double bv = 6.0;

  double alpha = 0.01;
  double kappa = 0.5;

  Index nsteps = tstar/tau;

  double ts_ee = tau / nsteps_ee;

  double hx = (bx-ax) / Nx;
  double hv = (bv-av) / Nv;

  vector<double*> X, V;

  multi_array<double,1> xx({Nx});

  for(Index i = 0; i < Nx; i++){
    xx(i) = 1.0 + alpha*cos(kappa*(ax + i*hx));
  }
  X.push_back(xx.begin());

  multi_array<double,1> v({Nv});
  multi_array<double,1> vv({Nv});

  for(Index i = 0; i < Nv; i++){
    v(i) = av + i*hv;
    vv(i) = (1.0/sqrt(2*M_PI)) *exp(-pow((av + i*hv),2)/2.0);
  }
  V.push_back(vv.begin());

  lr2<double> lr_sol(r,{Nx,Nv});

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({Nx});
  multi_array<complex<double>,1> efhat({Nx/2 + 1});

  array<fftw_plan,2> plans_e = create_plans_1d(Nx, ef, efhat);

  // For FFT -- Pay attention we have to cast to int as Index seems not to work with fftw_many

  multi_array<complex<double>,2> Khat({Nx/2+1,r});
  multi_array<complex<double>,2> Lhat({Nv/2+1,r});

  array<fftw_plan,2> plans_x = create_plans_1d(Nx, lr_sol.X, Khat);
  array<fftw_plan,2> plans_v = create_plans_1d(Nv, lr_sol.V, Lhat);

  std::function<double(double*,double*)> ip_x = inner_product_from_const_weight(hx, Nx);
  std::function<double(double*,double*)> ip_v = inner_product_from_const_weight(hv, Nv);

  // Mandatory to initialize after plan creation if we use FFTW_MEASURE
  initialize(lr_sol, X, V, ip_x, ip_v);

  multi_array<complex<double>,2> Kehat({Nx/2+1,r});
  multi_array<complex<double>,2> Mhattmp({Nx/2+1,r});

  multi_array<complex<double>,2> Lvhat({Nv/2+1,r});
  multi_array<complex<double>,2> Nhattmp({Nv/2+1,r});

  multi_array<complex<double>,1> lambdax({Nx/2+1});
  multi_array<complex<double>,1> lambdax_n({Nx/2+1});

  double ncx = 1.0 / Nx;

  for(Index j = 0; j < (Nx/2 + 1) ; j++){
    lambdax_n(j) = complex<double>(0.0,2.0*M_PI/(bx-ax)*j*ncx);
  }

  multi_array<complex<double>,1> lambdav({Nv/2+1});
  multi_array<complex<double>,1> lambdav_n({Nv/2+1});

  double ncv = 1.0 / Nv;

  for(Index j = 0; j < (Nv/2 + 1) ; j++){
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

  multi_array<double,2> T({r,r});
  multi_array<double,2> multmp({r,r});

  multi_array<complex<double>,2> Mhat({Nx/2 + 1,r});
  multi_array<complex<double>,2> Nhat({Nv/2 + 1,r});
  multi_array<complex<double>,2> Tc({r,r});

  int lwork = -1;
  schur(T, T, dc_r, lwork);

  // For D coefficients

  multi_array<double,2> D1({r,r});
  multi_array<double,2> D2({r,r});

  multi_array<double,1> wx({Nx});
  multi_array<double,2> dX({Nx,r});

  // For plots
  multi_array<double,1> int_x({r});
  multi_array<double,1> int_v({r});
  multi_array<double,1> tmp_vec({r});

  double mass0 = 0.0;
  double mass = 0.0;
  double energy0 = 0.0;
  double energy = 0.0;
  double el_energy = 0.0;
  double err_mass = 0.0;
  double err_energy = 0.0;


  // Initial mass
  coeff_one(lr_sol.X,hx,int_x);
  coeff_one(lr_sol.V,hv,int_v);

  matvec(lr_sol.S,int_v,tmp_vec);

  for(int ii = 0; ii < r; ii++){
    mass0 += (int_x(ii)*tmp_vec(ii));
  }

  // Initial energy

  coeff_one(lr_sol.V,hv,rho);

  rho *= -1.0;

  matvec(lr_sol.X,rho,ef);

  for(Index ii = 0; ii < Nx; ii++){
    ef(ii) += 1.0;
  }

  fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

  efhat(0) = complex<double>(0.0,0.0);
  for(Index ii = 1; ii < (Nx/2+1); ii++){
    complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*ii);

    efhat(ii) /= (lambdax/ncx);
  }

  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhat.begin(),ef.begin());
  energy0 = 0.0;
  for(Index ii = 0; ii < Nx; ii++){
    energy0 += 0.5*pow(ef(ii),2)*hx;
  }

  multi_array<double,1> wv2({Nv});

  for(Index j = 0; j < Nv; j++){
    wv2(j) = pow(v(j),2) * hv;
  }

  coeff_one(lr_sol.V,wv2.begin(),int_v);

  matvec(lr_sol.S,int_v,tmp_vec);

  for(int ii = 0; ii < r; ii++){
    energy0 += (0.5*int_x(ii)*tmp_vec(ii));
  }

  cout.precision(15);
  cout << std::scientific;

  ofstream el_energyf;
  ofstream err_massf;
  ofstream err_energyf;

  el_energyf.open("el_energy_1d.txt");
  err_massf.open("err_mass_1d.txt");
  err_energyf.open("err_energy_1d.txt");

  el_energyf.precision(16);
  err_massf.precision(16);
  err_energyf.precision(16);

  el_energyf << tstar << endl;
  el_energyf << tau << endl;

  multi_array<double,2> tmpX({Nx,r});
  multi_array<double,2> tmpV({Nv,r});
  multi_array<double,2> tmpS({r,r});

  multi_array<double,2> X0(lr_sol.X);
  multi_array<double,2> V0(lr_sol.V);
  multi_array<double,2> S0(lr_sol.S);

  multi_array<double,1> error_vec({6});


  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i << " on " << nsteps << endl;

    /* K step */
    tmpX = lr_sol.X;

    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field

    coeff_one(lr_sol.V,hv,rho);

    rho *= -1.0;

    matvec(lr_sol.X,rho,ef);

    for(Index ii = 0; ii < Nx; ii++){
      ef(ii) += 1.0;
    }

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    efhat(0) = complex<double>(0.0,0.0);
    for(Index ii = 1; ii < (Nx/2+1); ii++){
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*ii);

      efhat(ii) /= (lambdax/ncx);
    }

    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhat.begin(),ef.begin());

    // Main of K step

    coeff(lr_sol.V, lr_sol.V, wv.begin(), C1);

    fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

    ptw_mult_row(Lhat,lambdav_n.begin(),Lhat);

    fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),dV.begin());

    coeff(lr_sol.V, dV, hv, C2);

    schur(C1, T, dc_r, lwork);

    T.to_cplx(Tc);
    C2.to_cplx(C2c);

    fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    matmul(Khat,Tc,Mhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row(lr_sol.X,ef.begin(),lr_sol.X);

      fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Kehat.begin());

      matmul_transb(Kehat,C2c,Khat);
      matmul(Khat,Tc,Mhattmp);

      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

          Mhat(j,k) *= exp(-ts_ee*lambdax*dc_r(k));
          Mhat(j,k) += ts_ee*phi1_im(-ts_ee*lambdax*dc_r(k))*Mhattmp(j,k);
        }
      }
      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_x);

    /* S step */
    for(Index ii = 0; ii < Nx; ii++){
      wx(ii) = hx*ef(ii);
    }

    coeff(lr_sol.X, lr_sol.X, wx.begin(), D1);

    fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    ptw_mult_row(Khat,lambdax_n.begin(),Khat);

    fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),dX.begin());

    coeff(lr_sol.X, dX, hx, D2);

    // Explicit Euler
    for(int j = 0; j< nsteps_ee; j++){
      matmul_transb(lr_sol.S,C1,tmpS);
      matmul(D2,tmpS,T);

      matmul_transb(lr_sol.S,C2,tmpS);
      matmul(D1,tmpS,multmp);

      T -= multmp;
      T *= ts_ee;
      lr_sol.S += T;
    }

    /* L step */

    tmpV = lr_sol.V;
    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    schur(D1, T, dc_r, lwork);

    T.to_cplx(Tc);
    D2.to_cplx(C2c);

    fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

    matmul(Lhat,Tc,Nhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row(lr_sol.V,v.begin(),lr_sol.V);

      fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lvhat.begin());

      matmul_transb(Lvhat,C2c,Lhat);
      matmul(Lhat,Tc,Nhattmp);

      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nv/2 + 1); j++){
          complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(bv-av)*j);

          Nhat(j,k) *= exp(ts_ee*lambdav*dc_r(k));
          Nhat(j,k) += -ts_ee*phi1_im(ts_ee*lambdav*dc_r(k))*Nhattmp(j,k);
        }
      }
      matmul_transb(Nhat,Tc,Lhat);

      Lhat *= ncv;
      fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_v);

    transpose_inplace(lr_sol.S);

    // Electric energy
    el_energy = 0.0;
    for(Index ii = 0; ii < Nx; ii++){
      el_energy += 0.5*pow(ef(ii),2)*hx;
    }

    cout << "Electric energy: " << el_energy << endl;
    el_energyf << el_energy << endl;

    // Error mass
    coeff_one(lr_sol.X,hx,int_x);
    coeff_one(lr_sol.V,hv,int_v);

    matvec(lr_sol.S,int_v,tmp_vec);

    mass = 0.0;
    for(int ii = 0; ii < r; ii++){
      mass += (int_x(ii)*tmp_vec(ii));
    }

    err_mass = abs(mass0-mass);

    cout << "Error in mass: " << err_mass << endl;
    err_massf << err_mass << endl;

    coeff_one(lr_sol.V,wv2.begin(),int_v);

    matvec(lr_sol.S,int_v,tmp_vec);

    energy = el_energy;
    for(int ii = 0; ii < r; ii++){
      energy += (0.5*int_x(ii)*tmp_vec(ii));
    }

    err_energy = abs(energy0-energy);

    cout << "Error in energy: " << err_energy << endl;
    err_energyf << err_energy << endl;


  }

  el_energyf.close();
  err_massf.close();
  err_energyf.close();

  multi_array<double,2> refsol({Nx,Nv});
  multi_array<double,2> tmpsol({Nx,r});

  matmul(lr_sol.X,lr_sol.S,tmpsol);
  matmul_transb(tmpsol,lr_sol.V,refsol);

  for(double tau = pow(2,-3); tau > pow(2,-9); tau /= 2.0){
    lr_sol.X = X0;
    lr_sol.S = S0;
    lr_sol.V = V0;

    Index nsteps = tstar/tau;

    double ts_ee = tau / nsteps_ee;


      for(Index i = 0; i < nsteps; i++){

        cout << "Time step " << i << " on " << nsteps << endl;

        /* K step */
        tmpX = lr_sol.X;

        matmul(tmpX,lr_sol.S,lr_sol.X);

        // Electric field

        coeff_one(lr_sol.V,hv,rho);

        rho *= -1.0;

        matvec(lr_sol.X,rho,ef);

        for(Index ii = 0; ii < Nx; ii++){
          ef(ii) += 1.0;
        }

        fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

        efhat(0) = complex<double>(0.0,0.0);
        for(Index ii = 1; ii < (Nx/2+1); ii++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*ii);

          efhat(ii) /= (lambdax/ncx);
        }

        fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhat.begin(),ef.begin());

        // Main of K step

        coeff(lr_sol.V, lr_sol.V, wv.begin(), C1);

        fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

        ptw_mult_row(Lhat,lambdav_n.begin(),Lhat);

        fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),dV.begin());

        coeff(lr_sol.V, dV, hv, C2);

        schur(C1, T, dc_r, lwork);

        T.to_cplx(Tc);
        C2.to_cplx(C2c);

        fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

        matmul(Khat,Tc,Mhat);

        for(int kk = 0; kk < nsteps_ee; kk++){

          ptw_mult_row(lr_sol.X,ef.begin(),lr_sol.X);

          fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Kehat.begin());

          matmul_transb(Kehat,C2c,Khat);
          matmul(Khat,Tc,Mhattmp);

          for(int k = 0; k < r; k++){
            for(Index j = 0; j < (Nx/2 + 1); j++){
              complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

              Mhat(j,k) *= exp(-ts_ee*lambdax*dc_r(k));
              Mhat(j,k) += ts_ee*phi1_im(-ts_ee*lambdax*dc_r(k))*Mhattmp(j,k);
            }
          }
          matmul_transb(Mhat,Tc,Khat);

          Khat *= ncx;

          fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

        }

        gram_schmidt(lr_sol.X, lr_sol.S, ip_x);

        /* S step */
        for(Index ii = 0; ii < Nx; ii++){
          wx(ii) = hx*ef(ii);
        }

        coeff(lr_sol.X, lr_sol.X, wx.begin(), D1);

        fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

        ptw_mult_row(Khat,lambdax_n.begin(),Khat);

        fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),dX.begin());

        coeff(lr_sol.X, dX, hx, D2);

        // Explicit Euler
        for(int j = 0; j< nsteps_ee; j++){
          matmul_transb(lr_sol.S,C1,tmpS);
          matmul(D2,tmpS,T);

          matmul_transb(lr_sol.S,C2,tmpS);
          matmul(D1,tmpS,multmp);

          T -= multmp;
          T *= ts_ee;
          lr_sol.S += T;
        }

        /* L step */

        tmpV = lr_sol.V;
        matmul_transb(tmpV,lr_sol.S,lr_sol.V);

        schur(D1, T, dc_r, lwork);

        T.to_cplx(Tc);
        D2.to_cplx(C2c);

        fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

        matmul(Lhat,Tc,Nhat);

        for(int kk = 0; kk < nsteps_ee; kk++){

          ptw_mult_row(lr_sol.V,v.begin(),lr_sol.V);

          fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lvhat.begin());

          matmul_transb(Lvhat,C2c,Lhat);
          matmul(Lhat,Tc,Nhattmp);

          for(int k = 0; k < r; k++){
            for(Index j = 0; j < (Nv/2 + 1); j++){
              complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(bv-av)*j);

              Nhat(j,k) *= exp(ts_ee*lambdav*dc_r(k));
              Nhat(j,k) += -ts_ee*phi1_im(ts_ee*lambdav*dc_r(k))*Nhattmp(j,k);
            }
          }
          matmul_transb(Nhat,Tc,Lhat);

          Lhat *= ncv;
          fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

        }

        gram_schmidt(lr_sol.V, lr_sol.S, ip_v);

        transpose_inplace(lr_sol.S);

        // Electric energy
        el_energy = 0.0;
        for(Index ii = 0; ii < Nx; ii++){
          el_energy += 0.5*pow(ef(ii),2)*hx;
        }

        cout << "Electric energy: " << el_energy << endl;
        //el_energyf << el_energy << endl;

        // Error mass
        coeff_one(lr_sol.X,hx,int_x);
        coeff_one(lr_sol.V,hv,int_v);

        matvec(lr_sol.S,int_v,tmp_vec);

        mass = 0.0;
        for(int ii = 0; ii < r; ii++){
          mass += (int_x(ii)*tmp_vec(ii));
        }

        err_mass = abs(mass0-mass);

        cout << "Error in mass: " << err_mass << endl;
        //err_massf << err_mass << endl;

        coeff_one(lr_sol.V,wv2.begin(),int_v);

        matvec(lr_sol.S,int_v,tmp_vec);

        energy = el_energy;
        for(int ii = 0; ii < r; ii++){
          energy += (0.5*int_x(ii)*tmp_vec(ii));
        }

        err_energy = abs(energy0-energy);

        cout << "Error in energy: " << err_energy << endl;
        //err_energyf << err_energy << endl;


      }

      multi_array<double,2> sol({Nx,Nv});
      multi_array<double,2> tmp2sol({Nx,r});

      matmul(lr_sol.X,lr_sol.S,tmp2sol);
      matmul_transb(tmp2sol,lr_sol.V,sol);

      double error = 0.0;
      for(int iii = 0; iii < Nx; iii++){
        for(int jjj = 0; jjj < Nv; jjj++){
          if( error < abs(refsol(iii,jjj)-sol(iii,jjj))){
            error = abs(refsol(iii,jjj)-sol(iii,jjj));
          }
        }
      }
      error_vec(i_err) = error;

      time_error_1d << error << endl;
      tau_span_1d << tau << endl;

      i_err += 1;


  }

  cout << error_vec << endl;

  time_error_1d.close();
  tau_span_1d.close();
  return 0;
}
