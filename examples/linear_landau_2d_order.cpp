#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>

#include <random>
#include <complex>
#include <cstring>

int main(){

  ofstream time_error_2d;
  ofstream tau_span_2d;
  time_error_2d.open("time_error_2d.txt");
  tau_span_2d.open("tau_span_2d.txt");

  array<Index,2> N_xx = {16,16}; // Sizes in space
  array<Index,2> N_vv = {64,64}; // Sizes in velocity

  int r = 5; // rank desired

  double tstar = 10; // final time
  double tauref = pow(2,-10);
  double tau = tauref; // time step splitting
  int i_err = 0;


  array<double,4> lim_xx = {0.0,4.0*M_PI,0.0,4.0*M_PI}; // Limits for box [ax,bx] x [ay,by] {ax,bx,ay,by}
  array<double,4> lim_vv = {-6.0,6.0,-6.0,6.0}; // Limits for box [av,bv] x [aw,bw] {av,bv,aw,bw}

  double alpha = 0.01;
  double kappa1 = 0.5;
  double kappa2 = 0.5;

  Index nsteps_split = 1; // Number of time steps internal splitting
  Index nsteps_ee = 2; // Number of time steps of exponential euler in internal splitting

  Index nsteps = tstar/tau;

  double ts_split = tau / nsteps_split;
  double ts_ee = ts_split / nsteps_ee;

  array<double,2> h_xx, h_vv;
  int jj = 0;
  for(int ii = 0; ii < 2; ii++){
    h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
    h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
    jj+=2;
  }

  vector<double*> X, V;

  Index dxx_mult = N_xx[0]*N_xx[1];
  Index dxxh_mult = N_xx[1]*(N_xx[0]/2 + 1);

  Index dvv_mult = N_vv[0]*N_vv[1];
  Index dvvh_mult = N_vv[1]*(N_vv[0]/2 + 1);

  multi_array<double,1> xx({dxx_mult});

  for(Index j = 0; j < N_xx[1]; j++){
    for(Index i = 0; i < N_xx[0]; i++){
      double x = lim_xx[0] + i*h_xx[0];
      double y = lim_xx[2] + j*h_xx[1];
      xx(i+j*N_xx[0]) = 1.0 + alpha*cos(kappa1*x) + alpha*cos(kappa2*y);
    }
  }
  X.push_back(xx.begin());

  multi_array<double,1> v({dvv_mult});
  multi_array<double,1> w({dvv_mult});
  multi_array<double,1> vv({dvv_mult});

  for(Index j = 0; j < N_vv[1]; j++){
    for(Index i = 0; i < N_vv[0]; i++){
      v(i+j*N_vv[0]) = lim_vv[0] + i*h_vv[0];
      w(i+j*N_vv[0]) = lim_vv[2] + j*h_vv[1];
      vv(i+j*N_vv[0]) = (1.0/(2*M_PI)) *exp(-(pow(v(i+j*N_vv[0]),2)+pow(w(i+j*N_vv[0]),2))/2.0);
    }
  }
  V.push_back(vv.begin());

  lr2<double> lr_sol(r,{dxx_mult,dvv_mult});

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({dxx_mult});
  multi_array<double,1> efx({dxx_mult});
  multi_array<double,1> efy({dxx_mult});

  multi_array<complex<double>,1> efhat({dxxh_mult});
  multi_array<complex<double>,1> efhatx({dxxh_mult});
  multi_array<complex<double>,1> efhaty({dxxh_mult});

  array<fftw_plan,2> plans_e = create_plans_2d(N_xx, ef, efhat);

  // Some FFT stuff for X and V

  multi_array<complex<double>,1> lambdax_n({dxxh_mult});
  multi_array<complex<double>,1> lambday_n({dxxh_mult});

  double ncxx = 1.0 / (dxx_mult);

  Index mult;
  for(Index j = 0; j < N_xx[1]; j++){
    if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      lambdax_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i)*ncxx;
      lambday_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult)*ncxx;
    }
  }

  multi_array<complex<double>,1> lambdav_n({dvvh_mult});
  multi_array<complex<double>,1> lambdaw_n({dvvh_mult});

  double ncvv = 1.0 / (dvv_mult);

  for(Index j = 0; j < N_vv[1]; j++){
    if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
    for(Index i = 0; i < (N_vv[0]/2+1); i++){
      lambdav_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i)*ncvv;
      lambdaw_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult)*ncvv;
    }
  }

  multi_array<complex<double>,2> Khat({dxxh_mult,r});
  array<fftw_plan,2> plans_xx = create_plans_2d(N_xx, lr_sol.X, Khat);

  multi_array<complex<double>,2> Lhat({dvvh_mult,r});
  array<fftw_plan,2> plans_vv = create_plans_2d(N_vv, lr_sol.V, Lhat);

  // For C coefficients
  multi_array<double,2> C1v({r,r});
  multi_array<double,2> C1w({r,r});

  multi_array<double,2> C2v({r,r});
  multi_array<double,2> C2w({r,r});

  multi_array<complex<double>,2> C2vc({r,r});
  multi_array<complex<double>,2> C2wc({r,r});

  multi_array<double,1> we_v({dvv_mult});
  multi_array<double,1> we_w({dvv_mult});

  for(Index j = 0; j < (dvv_mult); j++){
    we_v(j) = v(j) * h_vv[0] * h_vv[1];
    we_w(j) = w(j) * h_vv[0] * h_vv[1];
  }

  multi_array<double,2> dV_v({dvv_mult,r});
  multi_array<double,2> dV_w({dvv_mult,r});

  multi_array<complex<double>,2> dVhat_v({dvvh_mult,r});
  multi_array<complex<double>,2> dVhat_w({dvvh_mult,r});

  // For D coefficients

  multi_array<double,2> D1x({r,r});
  multi_array<double,2> D1y({r,r});

  multi_array<double,2> D2x({r,r});
  multi_array<double,2> D2y({r,r});

  multi_array<double,1> we_x({dxx_mult});
  multi_array<double,1> we_y({dxx_mult});

  multi_array<double,2> dX_x({dxx_mult,r});
  multi_array<double,2> dX_y({dxx_mult,r});

  multi_array<complex<double>,2> dXhat_x({dxxh_mult,r});
  multi_array<complex<double>,2> dXhat_y({dxxh_mult,r});

  // For Schur decomposition
  multi_array<double,1> dcv_r({r});
  multi_array<double,1> dcw_r({r});

  multi_array<double,2> Tv({r,r});
  multi_array<double,2> Tw({r,r});

  multi_array<complex<double>,2> Mhat({dxxh_mult,r});
  multi_array<complex<double>,2> Nhat({dvvh_mult,r});
  multi_array<complex<double>,2> Tvc({r,r});
  multi_array<complex<double>,2> Twc({r,r});

  int lwork = -1;
  schur(Tv, Tv, dcv_r, lwork); // dumb call to obtain optimal value to work

  // For K step

  multi_array<double,2> Kex({dxx_mult,r});
  multi_array<double,2> Key({dxx_mult,r});

  multi_array<complex<double>,2> Kexhat({dxxh_mult,r});
  multi_array<complex<double>,2> Keyhat({dxxh_mult,r});

  // For L step
  multi_array<double,2> Lv({dvv_mult,r});
  multi_array<double,2> Lw({dvv_mult,r});

  multi_array<complex<double>,2> Lvhat({dvvh_mult,r});
  multi_array<complex<double>,2> Lwhat({dvvh_mult,r});

  // Temporary objects to perform multiplications
  multi_array<double,2> tmpX({dxx_mult,r});
  multi_array<double,2> tmpS({r,r});
  multi_array<double,2> tmpV({dvv_mult,r});

  multi_array<complex<double>,2> tmpXhat({dxxh_mult,r});
  multi_array<complex<double>,2> tmpVhat({dvvh_mult,r});

  // Quantities of interest
  multi_array<double,1> int_x({r});
  multi_array<double,1> int_v({r});
  multi_array<double,1> int_v2({r});

  double mass0 = 0.0;
  double mass = 0.0;
  double energy0 = 0.0;
  double energy = 0.0;
  double el_energy = 0.0;
  double err_mass = 0.0;
  double err_energy = 0.0;

  // Initialization
  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(h_xx[0]*h_xx[1], dxx_mult);
  std::function<double(double*,double*)> ip_vv = inner_product_from_const_weight(h_vv[0]*h_vv[1], dvv_mult);

  initialize(lr_sol, X, V, ip_xx, ip_vv); // Mandatory to initialize after plan creation as we use FFTW_MEASURE

  // Initial mass
  coeff_one(lr_sol.X,h_xx[0]*h_xx[1],int_x);
  coeff_one(lr_sol.V,h_vv[0]*h_vv[1],int_v);

  matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    mass0 += (int_x(ii)*rho(ii));
  }

  // Initial energy
  coeff_one(lr_sol.V,h_vv[0]*h_vv[1],rho);
  rho *= -1.0;
  matvec(lr_sol.X,rho,ef);
  for(Index ii = 0; ii < dxx_mult; ii++){
    ef(ii) += 1.0;
  }
  fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());
  for(Index j = 0; j < N_xx[1]; j++){
    if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);
      efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
      efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
    }
  }
  efhatx(0) = complex<double>(0.0,0.0);
  efhaty(0) = complex<double>(0.0,0.0);
  efhatx((N_xx[0]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
  efhaty((N_xx[0]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());

  energy0 = 0.0;
  for(Index ii = 0; ii < (dxx_mult); ii++){
    energy0 += 0.5*(pow(efx(ii),2)+pow(efy(ii),2))*h_xx[0]*h_xx[1];
  }

  multi_array<double,1> we_v2({dvv_mult});
  multi_array<double,1> we_w2({dvv_mult});

  for(Index j = 0; j < (dvv_mult); j++){
    we_v2(j) = pow(v(j),2) * h_vv[0] * h_vv[1];
    we_w2(j) = pow(w(j),2) * h_vv[0] * h_vv[1];
  }

  coeff_one(lr_sol.V,we_v2.begin(),int_v);
  coeff_one(lr_sol.V,we_w2.begin(),int_v2);

  int_v += int_v2;

  matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    energy0 += 0.5*(int_x(ii)*rho(ii));
  }

  cout.precision(15);
  cout << std::scientific;

  ofstream el_energyf;
  ofstream err_massf;
  ofstream err_energyf;

  el_energyf.open("el_energy_2d.txt");
  err_massf.open("err_mass_2d.txt");
  err_energyf.open("err_energy_2d.txt");

  el_energyf.precision(16);
  err_massf.precision(16);
  err_energyf.precision(16);

  el_energyf << tstar << endl;
  el_energyf << tau << endl;

  multi_array<double,2> X0(lr_sol.X);
  multi_array<double,2> V0(lr_sol.V);
  multi_array<double,2> S0(lr_sol.S);

  multi_array<double,1> error_vec({6});


  // Main cycle
  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i + 1 << " on " << nsteps << endl;

    tmpX = lr_sol.X;
    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field

    coeff_one(lr_sol.V,h_vv[0]*h_vv[1],rho);
    rho *= -1.0;
    matvec(lr_sol.X,rho,ef);

    for(Index ii = 0; ii < dxx_mult; ii++){
      ef(ii) += 1.0;
    }

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    for(Index j = 0; j < N_xx[1]; j++){
      if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
        complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

        efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
        efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
      }
    }

    efhatx(0) = complex<double>(0.0,0.0);
    efhaty(0) = complex<double>(0.0,0.0);
    efhatx((N_xx[0]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
    efhaty((N_xx[0]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());

    // Main of K step

    coeff(lr_sol.V, lr_sol.V, we_v.begin(), C1v);
    coeff(lr_sol.V, lr_sol.V, we_w.begin(), C1w);

    fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)tmpVhat.begin());

    ptw_mult_row(tmpVhat,lambdav_n.begin(),dVhat_v);
    ptw_mult_row(tmpVhat,lambdaw_n.begin(),dVhat_w);

    fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_v.begin(),dV_v.begin());
    fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_w.begin(),dV_w.begin());

    coeff(lr_sol.V, dV_v, h_vv[0]*h_vv[1], C2v);
    coeff(lr_sol.V, dV_w, h_vv[0]*h_vv[1], C2w);

    schur(C1v, Tv, dcv_r, lwork);
    schur(C1w, Tw, dcw_r, lwork);

    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    C2v.to_cplx(C2vc);
    C2w.to_cplx(C2wc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution
      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());
      matmul(Khat,Tvc,Mhat);

      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
            complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
            Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-ts_split*lambdax*dcv_r(k))*ncxx;
          }
        }
      }

      matmul_transb(Mhat,Tvc,Khat);
      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin()); // TODO: could use FFTW_PRESERVE_INPUT here
      // NOT POSSIBLE FOR multidimensional transforms, see documentation

      // Full step -- Exponential Euler
      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());
      matmul(Khat,Twc,Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.X,efx.begin(),Kex);
        ptw_mult_row(lr_sol.X,efy.begin(),Key);

        fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
        fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());

        matmul_transb(Kexhat,C2vc,Khat);
        matmul_transb(Keyhat,C2wc,tmpXhat);

        Khat += tmpXhat;

        matmul(Khat,Twc,tmpXhat);

        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-ts_ee*lambday*dcw_r(k));
              Mhat(i+j*(N_xx[0]/2+1),k) += ts_ee*phi1_im(-ts_ee*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }

        matmul_transb(Mhat,Twc,Khat);
        Khat *= ncxx;
        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());
      }
    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_xx);

    // S Step

    for(Index j = 0; j < (dxx_mult); j++){
      we_x(j) = efx(j) * h_xx[0] * h_xx[1];
      we_y(j) = efy(j) * h_xx[0] * h_xx[1];
    }

    coeff(lr_sol.X, lr_sol.X, we_x.begin(), D1x);
    coeff(lr_sol.X, lr_sol.X, we_y.begin(), D1y);

    fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)tmpXhat.begin());

    ptw_mult_row(tmpXhat,lambdax_n.begin(),dXhat_x);
    ptw_mult_row(tmpXhat,lambday_n.begin(),dXhat_y);

    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_x.begin(),dX_x.begin());
    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_y.begin(),dX_y.begin());

    coeff(lr_sol.X, dX_x, h_xx[0]*h_xx[1], D2x);
    coeff(lr_sol.X, dX_y, h_xx[0]*h_xx[1], D2y);

    // Explicit Euler
    for(Index jj = 0; jj< nsteps_split; jj++){
      matmul_transb(lr_sol.S,C1v,tmpS);
      matmul(D2x,tmpS,Tv);

      matmul_transb(lr_sol.S,C1w,tmpS);
      matmul(D2y,tmpS,Tw);

      Tv += Tw;

      matmul_transb(lr_sol.S,C2v,tmpS);
      matmul(D1x,tmpS,Tw);

      Tv -= Tw;

      matmul_transb(lr_sol.S,C2w,tmpS);
      matmul(D1y,tmpS,Tw);

      Tv -= Tw;

      Tv *= ts_split;
      lr_sol.S += Tv;
    }

    // L step - here we reuse some old variable names

    tmpV = lr_sol.V;

    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    schur(D1x, Tv, dcv_r, lwork);
    schur(D1y, Tw, dcw_r, lwork);

    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    D2x.to_cplx(C2vc);
    D2y.to_cplx(C2wc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());
      matmul(Lhat,Tvc,Nhat);

      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_vv[1]; j++){
          for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
            complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
            Nhat(i+j*(N_vv[0]/2+1),k) *= exp(ts_split*lambdav*dcv_r(k))*ncvv;
          }
        }
      }

      matmul_transb(Nhat,Tvc,Lhat);
      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      // Full step -- Exponential euler
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin()); // TODO: use FFTW_PRESERVE_INPUT
      // NOT POSSIBLE FOR multidimensional transforms, see documentation

      matmul(Lhat,Twc,Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.V,v.begin(),Lv);
        ptw_mult_row(lr_sol.V,w.begin(),Lw);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());

        matmul_transb(Lvhat,C2vc,Lhat);
        matmul_transb(Lwhat,C2wc,tmpVhat);

        Lhat += tmpVhat;

        matmul(Lhat,Twc,tmpVhat);

        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);
              Nhat(i+j*(N_vv[0]/2+1),k) *= exp(ts_ee*lambdaw*dcw_r(k));
              Nhat(i+j*(N_vv[0]/2+1),k) -= ts_ee*phi1_im(ts_ee*lambdaw*dcw_r(k))*tmpVhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }

        matmul_transb(Nhat,Twc,Lhat);
        Lhat *= ncvv;
        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());
      }

    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_vv);

    transpose_inplace(lr_sol.S);

    // Electric energy
    el_energy = 0.0;
    for(Index ii = 0; ii < (dxx_mult); ii++){
      el_energy += 0.5*(pow(efx(ii),2)+pow(efy(ii),2))*h_xx[0]*h_xx[1];
    }
    cout << "Electric energy: " << el_energy << endl;
    el_energyf << el_energy << endl;

    // Error Mass
    coeff_one(lr_sol.X,h_xx[0]*h_xx[1],int_x);
    coeff_one(lr_sol.V,h_vv[0]*h_vv[1],int_v);

    matvec(lr_sol.S,int_v,rho);

    mass = 0.0;
    for(int ii = 0; ii < r; ii++){
      mass += (int_x(ii)*rho(ii));
    }

    err_mass = abs(mass0-mass);

    cout << "Error in mass: " << err_mass << endl;
    err_massf << err_mass << endl;

    coeff_one(lr_sol.V,we_v2.begin(),int_v);
    coeff_one(lr_sol.V,we_w2.begin(),int_v2);

    int_v += int_v2;

    matvec(lr_sol.S,int_v,rho);

    energy = el_energy;
    for(int ii = 0; ii < r; ii++){
      energy += 0.5*(int_x(ii)*rho(ii));
    }

    err_energy = abs(energy0-energy);

    cout << "Error in energy: " << err_energy << endl;
    err_energyf << err_energy << endl;


  }

  el_energyf.close();
  err_massf.close();
  err_energyf.close();

  multi_array<double,2> refsol({dxx_mult,dvv_mult});
  multi_array<double,2> tmpsol({dxx_mult,r});

  matmul(lr_sol.X,lr_sol.S,tmpsol);
  matmul_transb(tmpsol,lr_sol.V,refsol);

  for(double tau = pow(2,-3); tau > pow(2,-9); tau /= 2.0){
    lr_sol.X = X0;
    lr_sol.S = S0;
    lr_sol.V = V0;

    Index nsteps = tstar/tau;

    double ts_split = tau / nsteps_split;
    double ts_ee = ts_split / nsteps_ee;


    for(Index i = 0; i < nsteps; i++){

      cout << "Time step " << i + 1 << " on " << nsteps << endl;

      tmpX = lr_sol.X;
      matmul(tmpX,lr_sol.S,lr_sol.X);

      // Electric field

      coeff_one(lr_sol.V,h_vv[0]*h_vv[1],rho);
      rho *= -1.0;
      matvec(lr_sol.X,rho,ef);

      for(Index ii = 0; ii < dxx_mult; ii++){
        ef(ii) += 1.0;
      }

      fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

      for(Index j = 0; j < N_xx[1]; j++){
        if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
        for(Index i = 0; i < (N_xx[0]/2+1); i++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
          complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

          efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
          efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
        }
      }

      efhatx(0) = complex<double>(0.0,0.0);
      efhaty(0) = complex<double>(0.0,0.0);
      efhatx((N_xx[0]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
      efhaty((N_xx[0]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

      fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
      fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());

      // Main of K step

      coeff(lr_sol.V, lr_sol.V, we_v.begin(), C1v);
      coeff(lr_sol.V, lr_sol.V, we_w.begin(), C1w);

      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)tmpVhat.begin());

      ptw_mult_row(tmpVhat,lambdav_n.begin(),dVhat_v);
      ptw_mult_row(tmpVhat,lambdaw_n.begin(),dVhat_w);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_v.begin(),dV_v.begin());
      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_w.begin(),dV_w.begin());

      coeff(lr_sol.V, dV_v, h_vv[0]*h_vv[1], C2v);
      coeff(lr_sol.V, dV_w, h_vv[0]*h_vv[1], C2w);

      schur(C1v, Tv, dcv_r, lwork);
      schur(C1w, Tw, dcw_r, lwork);

      Tv.to_cplx(Tvc);
      Tw.to_cplx(Twc);
      C2v.to_cplx(C2vc);
      C2w.to_cplx(C2wc);

      // Internal splitting
      for(Index ii = 0; ii < nsteps_split; ii++){
        // Full step -- Exact solution
        fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());
        matmul(Khat,Tvc,Mhat);

        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-ts_split*lambdax*dcv_r(k))*ncxx;
            }
          }
        }

        matmul_transb(Mhat,Tvc,Khat);
        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin()); // TODO: could use FFTW_PRESERVE_INPUT here
        // NOT POSSIBLE FOR multidimensional transforms, see documentation

        // Full step -- Exponential Euler
        fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());
        matmul(Khat,Twc,Mhat);
        for(Index jj = 0; jj < nsteps_ee; jj++){

          ptw_mult_row(lr_sol.X,efx.begin(),Kex);
          ptw_mult_row(lr_sol.X,efy.begin(),Key);

          fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());

          matmul_transb(Kexhat,C2vc,Khat);
          matmul_transb(Keyhat,C2wc,tmpXhat);

          Khat += tmpXhat;

          matmul(Khat,Twc,tmpXhat);

          for(int k = 0; k < r; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

                Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-ts_ee*lambday*dcw_r(k));
                Mhat(i+j*(N_xx[0]/2+1),k) += ts_ee*phi1_im(-ts_ee*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
              }
            }
          }

          matmul_transb(Mhat,Twc,Khat);
          Khat *= ncxx;
          fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());
        }
      }

      gram_schmidt(lr_sol.X, lr_sol.S, ip_xx);

      // S Step

      for(Index j = 0; j < (dxx_mult); j++){
        we_x(j) = efx(j) * h_xx[0] * h_xx[1];
        we_y(j) = efy(j) * h_xx[0] * h_xx[1];
      }

      coeff(lr_sol.X, lr_sol.X, we_x.begin(), D1x);
      coeff(lr_sol.X, lr_sol.X, we_y.begin(), D1y);

      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)tmpXhat.begin());

      ptw_mult_row(tmpXhat,lambdax_n.begin(),dXhat_x);
      ptw_mult_row(tmpXhat,lambday_n.begin(),dXhat_y);

      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_x.begin(),dX_x.begin());
      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_y.begin(),dX_y.begin());

      coeff(lr_sol.X, dX_x, h_xx[0]*h_xx[1], D2x);
      coeff(lr_sol.X, dX_y, h_xx[0]*h_xx[1], D2y);

      // Explicit Euler
      for(Index jj = 0; jj< nsteps_split; jj++){
        matmul_transb(lr_sol.S,C1v,tmpS);
        matmul(D2x,tmpS,Tv);

        matmul_transb(lr_sol.S,C1w,tmpS);
        matmul(D2y,tmpS,Tw);

        Tv += Tw;

        matmul_transb(lr_sol.S,C2v,tmpS);
        matmul(D1x,tmpS,Tw);

        Tv -= Tw;

        matmul_transb(lr_sol.S,C2w,tmpS);
        matmul(D1y,tmpS,Tw);

        Tv -= Tw;

        Tv *= ts_split;
        lr_sol.S += Tv;
      }

      // L step - here we reuse some old variable names

      tmpV = lr_sol.V;

      matmul_transb(tmpV,lr_sol.S,lr_sol.V);

      schur(D1x, Tv, dcv_r, lwork);
      schur(D1y, Tw, dcw_r, lwork);

      Tv.to_cplx(Tvc);
      Tw.to_cplx(Twc);
      D2x.to_cplx(C2vc);
      D2y.to_cplx(C2wc);

      // Internal splitting
      for(Index ii = 0; ii < nsteps_split; ii++){
        // Full step -- Exact solution
        fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());
        matmul(Lhat,Tvc,Nhat);

        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
              Nhat(i+j*(N_vv[0]/2+1),k) *= exp(ts_split*lambdav*dcv_r(k))*ncvv;
            }
          }
        }

        matmul_transb(Nhat,Tvc,Lhat);
        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

        // Full step -- Exponential euler
        fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin()); // TODO: use FFTW_PRESERVE_INPUT
        // NOT POSSIBLE FOR multidimensional transforms, see documentation

        matmul(Lhat,Twc,Nhat);
        for(Index jj = 0; jj < nsteps_ee; jj++){

          ptw_mult_row(lr_sol.V,v.begin(),Lv);
          ptw_mult_row(lr_sol.V,w.begin(),Lw);

          fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());

          matmul_transb(Lvhat,C2vc,Lhat);
          matmul_transb(Lwhat,C2wc,tmpVhat);

          Lhat += tmpVhat;

          matmul(Lhat,Twc,tmpVhat);

          for(int k = 0; k < r; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);
                Nhat(i+j*(N_vv[0]/2+1),k) *= exp(ts_ee*lambdaw*dcw_r(k));
                Nhat(i+j*(N_vv[0]/2+1),k) -= ts_ee*phi1_im(ts_ee*lambdaw*dcw_r(k))*tmpVhat(i+j*(N_vv[0]/2+1),k);
              }
            }
          }

          matmul_transb(Nhat,Twc,Lhat);
          Lhat *= ncvv;
          fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());
        }

      }

      gram_schmidt(lr_sol.V, lr_sol.S, ip_vv);

      transpose_inplace(lr_sol.S);

      // Electric energy
      el_energy = 0.0;
      for(Index ii = 0; ii < (dxx_mult); ii++){
        el_energy += 0.5*(pow(efx(ii),2)+pow(efy(ii),2))*h_xx[0]*h_xx[1];
      }
      cout << "Electric energy: " << el_energy << endl;
      //el_energyf << el_energy << endl;

      // Error Mass
      coeff_one(lr_sol.X,h_xx[0]*h_xx[1],int_x);
      coeff_one(lr_sol.V,h_vv[0]*h_vv[1],int_v);

      matvec(lr_sol.S,int_v,rho);

      mass = 0.0;
      for(int ii = 0; ii < r; ii++){
        mass += (int_x(ii)*rho(ii));
      }

      err_mass = abs(mass0-mass);

      cout << "Error in mass: " << err_mass << endl;
      //err_massf << err_mass << endl;

      coeff_one(lr_sol.V,we_v2.begin(),int_v);
      coeff_one(lr_sol.V,we_w2.begin(),int_v2);

      int_v += int_v2;

      matvec(lr_sol.S,int_v,rho);

      energy = el_energy;
      for(int ii = 0; ii < r; ii++){
        energy += 0.5*(int_x(ii)*rho(ii));
      }

      err_energy = abs(energy0-energy);

      cout << "Error in energy: " << err_energy << endl;
      //err_energyf << err_energy << endl;


    }

    multi_array<double,2> sol({dxx_mult,dvv_mult});
    multi_array<double,2> tmp2sol({dxx_mult,r});

    matmul(lr_sol.X,lr_sol.S,tmp2sol);
    matmul_transb(tmp2sol,lr_sol.V,sol);

    double error = 0.0;
    for(int iii = 0; iii < dxx_mult; iii++){
      for(int jjj = 0; jjj < dvv_mult; jjj++){
        if( error < abs(refsol(iii,jjj)-sol(iii,jjj))){
          error = abs(refsol(iii,jjj)-sol(iii,jjj));
        }
      }
    }
    error_vec(i_err) = error;

    time_error_2d << error << endl;
    tau_span_2d << tau << endl;

    i_err += 1;


  }

  cout << error_vec << endl;

  time_error_2d.close();
  tau_span_2d.close();

  return 0;
}
