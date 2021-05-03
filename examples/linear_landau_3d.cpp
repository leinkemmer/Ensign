#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/timer.hpp>

#include <random>
#include <complex>
#include <cstring>

int main(){

  array<Index,3> N_xx = {16,16,16}; // Sizes in space
  array<Index,3> N_vv = {32,32,32}; // Sizes in velocity
  int r = 10; // rank desired

  double tstar = 20; // final time
  double tau = 0.00625; // time step splitting

  array<double,6> lim_xx = {0.0,4.0*M_PI,0.0,4.0*M_PI,0.0,4.0*M_PI}; // Limits for box [ax,bx] x [ay,by] x [az,bz] {ax,bx,ay,by,az,bz}
  array<double,6> lim_vv = {-6.0,6.0,-6.0,6.0,-6.0,6.0}; // Limits for box [av,bv] x [aw,bw] x [au,bu] {av,bv,aw,bw,au,bu}

  double alpha = 0.01;
  double kappa1 = 0.5;
  double kappa2 = 0.5;
  double kappa3 = 0.5;

  Index nsteps_split = 1; // Number of time steps internal splitting
  Index nsteps_ee = 1; // Number of time steps of exponential euler in internal splitting

  Index nsteps = tstar/tau;

  double ts_split = tau / nsteps_split;
  double ts_ee = ts_split / nsteps_ee;

  array<double,3> h_xx, h_vv;
  int jj = 0;
  for(int ii = 0; ii < 3; ii++){
    h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
    h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
    jj+=2;
  }

  vector<double*> X, V;

  Index dxx_mult = N_xx[0]*N_xx[1]*N_xx[2];
  Index dxxh_mult = N_xx[2]*N_xx[1]*(N_xx[0]/2 + 1);

  Index dvv_mult = N_vv[0]*N_vv[1]*N_vv[2];
  Index dvvh_mult = N_vv[2]*N_vv[1]*(N_vv[0]/2 + 1);

  multi_array<double,1> xx({dxx_mult});

  for(Index k = 0; k < N_xx[2]; k++){
    for(Index j = 0; j < N_xx[1]; j++){
      for(Index i = 0; i < N_xx[0]; i++){
        double x = lim_xx[0] + i*h_xx[0];
        double y = lim_xx[2] + j*h_xx[1];
        double z = lim_xx[4] + k*h_xx[2];

        xx(i+j*N_xx[0] + k*(N_xx[0]*N_xx[1])) = 1.0 + alpha*cos(kappa1*x) + alpha*cos(kappa2*y) + alpha*cos(kappa3*z);
      }
    }
  }
  X.push_back(xx.begin());

  multi_array<double,1> v({dvv_mult});
  multi_array<double,1> w({dvv_mult});
  multi_array<double,1> u({dvv_mult});

  multi_array<double,1> vv({dvv_mult});

  for(Index k = 0; k < N_vv[2]; k++){
    for(Index j = 0; j < N_vv[1]; j++){
      for(Index i = 0; i < N_vv[0]; i++){
        Index idx = i+j*N_vv[0] + k*(N_vv[0]*N_vv[1]);
        v(idx) = lim_vv[0] + i*h_vv[0];
        w(idx) = lim_vv[2] + j*h_vv[1];
        u(idx) = lim_vv[4] + k*h_vv[2];

        vv(idx) = (1.0/(sqrt(pow(2*M_PI,3)))) * exp(-(pow(v(idx),2)+pow(w(idx),2)+pow(u(idx),2))/2.0);
      }
    }
  }
  V.push_back(vv.begin());

  lr2<double> lr_sol(r,{dxx_mult,dvv_mult});

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({dxx_mult});
  multi_array<double,1> efx({dxx_mult});
  multi_array<double,1> efy({dxx_mult});
  multi_array<double,1> efz({dxx_mult});

  multi_array<complex<double>,1> efhat({dxxh_mult});
  multi_array<complex<double>,1> efhatx({dxxh_mult});
  multi_array<complex<double>,1> efhaty({dxxh_mult});
  multi_array<complex<double>,1> efhatz({dxxh_mult});

  array<fftw_plan,2> plans_e = create_plans_3d(N_xx, ef, efhat);

  // Some FFT stuff for X and V

  multi_array<complex<double>,1> lambdax_n({dxxh_mult});
  multi_array<complex<double>,1> lambday_n({dxxh_mult});
  multi_array<complex<double>,1> lambdaz_n({dxxh_mult});

  double ncxx = 1.0 / (dxx_mult);

  Index mult_j;
  Index mult_k;

  for(Index k = 0; k < N_xx[2]; k++){
    if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
    for(Index j = 0; j < N_xx[1]; j++){
      if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
        lambdax_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i)*ncxx;
        lambday_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j)*ncxx;
        lambdaz_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k)*ncxx;
      }
    }
  }

  multi_array<complex<double>,1> lambdav_n({dvvh_mult});
  multi_array<complex<double>,1> lambdaw_n({dvvh_mult});
  multi_array<complex<double>,1> lambdau_n({dvvh_mult});

  double ncvv = 1.0 / (dvv_mult);

  for(Index k = 0; k < N_vv[2]; k++){
    if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
    for(Index j = 0; j < N_vv[1]; j++){
      if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
      for(Index i = 0; i < (N_vv[0]/2+1); i++){
        Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

        lambdav_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i)*ncvv;
        lambdaw_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j)*ncvv;
        lambdau_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k)*ncvv;
      }
    }
  }

  multi_array<complex<double>,2> Khat({dxxh_mult,r});
  array<fftw_plan,2> plans_xx = create_plans_3d(N_xx, lr_sol.X, Khat);

  multi_array<complex<double>,2> Lhat({dvvh_mult,r});
  array<fftw_plan,2> plans_vv = create_plans_3d(N_vv, lr_sol.V, Lhat);

  // For C coefficients
  multi_array<double,2> C1v({r,r});
  multi_array<double,2> C1w({r,r});
  multi_array<double,2> C1u({r,r});

  multi_array<double,2> C2v({r,r});
  multi_array<double,2> C2w({r,r});
  multi_array<double,2> C2u({r,r});

  multi_array<complex<double>,2> C2vc({r,r});
  multi_array<complex<double>,2> C2wc({r,r});
  multi_array<complex<double>,2> C2uc({r,r});

  multi_array<double,1> we_v({dvv_mult});
  multi_array<double,1> we_w({dvv_mult});
  multi_array<double,1> we_u({dvv_mult});

  for(Index j = 0; j < (dvv_mult); j++){
    we_v(j) = v(j) * h_vv[0] * h_vv[1] * h_vv[2];
    we_w(j) = w(j) * h_vv[0] * h_vv[1] * h_vv[2];
    we_u(j) = u(j) * h_vv[0] * h_vv[1] * h_vv[2];
  }

  multi_array<double,2> dV_v({dvv_mult,r});
  multi_array<double,2> dV_w({dvv_mult,r});
  multi_array<double,2> dV_u({dvv_mult,r});

  multi_array<complex<double>,2> dVhat_v({dvvh_mult,r});
  multi_array<complex<double>,2> dVhat_w({dvvh_mult,r});
  multi_array<complex<double>,2> dVhat_u({dvvh_mult,r});

  // For D coefficients

  multi_array<double,2> D1x({r,r});
  multi_array<double,2> D1y({r,r});
  multi_array<double,2> D1z({r,r});

  multi_array<double,2> D2x({r,r});
  multi_array<double,2> D2y({r,r});
  multi_array<double,2> D2z({r,r});

  multi_array<double,1> we_x({dxx_mult});
  multi_array<double,1> we_y({dxx_mult});
  multi_array<double,1> we_z({dxx_mult});

  multi_array<double,2> dX_x({dxx_mult,r});
  multi_array<double,2> dX_y({dxx_mult,r});
  multi_array<double,2> dX_z({dxx_mult,r});

  multi_array<complex<double>,2> dXhat_x({dxxh_mult,r});
  multi_array<complex<double>,2> dXhat_y({dxxh_mult,r});
  multi_array<complex<double>,2> dXhat_z({dxxh_mult,r});

  // For Schur decomposition
  multi_array<double,1> dcv_r({r});
  multi_array<double,1> dcw_r({r});
  multi_array<double,1> dcu_r({r});

  multi_array<double,2> Tv({r,r});
  multi_array<double,2> Tw({r,r});
  multi_array<double,2> Tu({r,r});

  multi_array<complex<double>,2> Mhat({dxxh_mult,r});
  multi_array<complex<double>,2> Nhat({dvvh_mult,r});
  multi_array<complex<double>,2> Tvc({r,r});
  multi_array<complex<double>,2> Twc({r,r});
  multi_array<complex<double>,2> Tuc({r,r});

  int lwork = -1;
  schur(Tv, Tv, dcv_r, lwork); // dumb call to obtain optimal value to work

  // For K step

  multi_array<double,2> Kex({dxx_mult,r});
  multi_array<double,2> Key({dxx_mult,r});
  multi_array<double,2> Kez({dxx_mult,r});

  multi_array<complex<double>,2> Kexhat({dxxh_mult,r});
  multi_array<complex<double>,2> Keyhat({dxxh_mult,r});
  multi_array<complex<double>,2> Kezhat({dxxh_mult,r});

  // For L step
  multi_array<double,2> Lv({dvv_mult,r});
  multi_array<double,2> Lw({dvv_mult,r});
  multi_array<double,2> Lu({dvv_mult,r});

  multi_array<complex<double>,2> Lvhat({dvvh_mult,r});
  multi_array<complex<double>,2> Lwhat({dvvh_mult,r});
  multi_array<complex<double>,2> Luhat({dvvh_mult,r});

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
  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(h_xx[0]*h_xx[1]*h_xx[2], dxx_mult);
  std::function<double(double*,double*)> ip_vv = inner_product_from_const_weight(h_vv[0]*h_vv[1]*h_vv[2], dvv_mult);

  initialize(lr_sol, X, V, ip_xx, ip_vv); // Mandatory to initialize after plan creation as we use FFTW_MEASURE

  cout.precision(15);
  cout << std::scientific;

  // Initial mass
  coeff_one(lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],int_x);
  coeff_one(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],int_v);

  matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    mass0 += (int_x(ii)*rho(ii));
  }

  // Initial energy
  coeff_one(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],rho);
  rho *= -1.0;
  matvec(lr_sol.X,rho,ef);
  for(Index ii = 0; ii < dxx_mult; ii++){
    ef(ii) += 1.0;
  }
  fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());
  for(Index k = 0; k < N_xx[2]; k++){
    if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
    for(Index j = 0; j < N_xx[1]; j++){
      if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
        complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
        complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

        Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

        efhatx(idx) = efhat(idx) * lambdax / (pow(lambdax,2) + pow(lambday,2) + pow(lambdaz,2)) * ncxx;
        efhaty(idx) = efhat(idx) * lambday / (pow(lambdax,2) + pow(lambday,2) + pow(lambdaz,2)) * ncxx ;
        efhatz(idx) = efhat(idx) * lambdaz / (pow(lambdax,2) + pow(lambday,2) + pow(lambdaz,2)) * ncxx ;
      }
    }
  }
  for(Index k = 0; k < (N_xx[2]/2 + 1); k += (N_xx[2]/2)){
    for(Index j = 0; j < (N_xx[1]/2 + 1); j += (N_xx[1]/2)){
      efhatx(j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1])) = complex<double>(0.0,0.0);
      efhaty(j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1])) = complex<double>(0.0,0.0);
      efhatz(j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1])) = complex<double>(0.0,0.0);
    }
  }
  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());
  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatz.begin(),efz.begin());

  energy0 = 0.0;
  for(Index ii = 0; ii < (dxx_mult); ii++){
    energy0 += 0.5*(pow(efx(ii),2)+pow(efy(ii),2)+pow(efz(ii),2))*h_xx[0]*h_xx[1]*h_xx[2];
  }

  multi_array<double,1> we_v2({dvv_mult});
  multi_array<double,1> we_w2({dvv_mult});
  multi_array<double,1> we_u2({dvv_mult});

  for(Index j = 0; j < (dvv_mult); j++){
    we_v2(j) = pow(v(j),2) * h_vv[0] * h_vv[1] * h_vv[2];
    we_w2(j) = pow(w(j),2) * h_vv[0] * h_vv[1] * h_vv[2];
    we_u2(j) = pow(u(j),2) * h_vv[0] * h_vv[1] * h_vv[2];
  }

  coeff_one(lr_sol.V,we_v2.begin(),int_v);
  coeff_one(lr_sol.V,we_w2.begin(),int_v2);

  int_v += int_v2;

  coeff_one(lr_sol.V,we_u2.begin(),int_v2);

  int_v += int_v2;

  matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    energy0 += 0.5*(int_x(ii)*rho(ii));
  }

  ofstream el_energyf;
  ofstream err_massf;
  ofstream err_energyf;

  el_energyf.open("el_energy_3d.txt");
  err_massf.open("err_mass_3d.txt");
  err_energyf.open("err_energy_3d.txt");

  el_energyf.precision(16);
  err_massf.precision(16);
  err_energyf.precision(16);

  el_energyf << tstar << endl;
  el_energyf << tau << endl;

  gt::start("time_loop");
  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i + 1 << " on " << nsteps << endl;

    tmpX = lr_sol.X;
    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field
    gt::start("electric_field");

    coeff_one(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],rho);
    rho *= -1.0;
    matvec(lr_sol.X,rho,ef);

    for(Index ii = 0; ii < dxx_mult; ii++){
      ef(ii) += 1.0;
    }

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    for(Index k = 0; k < N_xx[2]; k++){
      if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
      for(Index j = 0; j < N_xx[1]; j++){
        if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
        for(Index i = 0; i < (N_xx[0]/2+1); i++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
          complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
          complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

          Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

          efhatx(idx) = efhat(idx) * lambdax / (pow(lambdax,2) + pow(lambday,2) + pow(lambdaz,2)) * ncxx;
          efhaty(idx) = efhat(idx) * lambday / (pow(lambdax,2) + pow(lambday,2) + pow(lambdaz,2)) * ncxx ;
          efhatz(idx) = efhat(idx) * lambdaz / (pow(lambdax,2) + pow(lambday,2) + pow(lambdaz,2)) * ncxx ;
        }
      }
    }
    for(Index k = 0; k < (N_xx[2]/2 + 1); k += (N_xx[2]/2)){
      for(Index j = 0; j < (N_xx[1]/2 + 1); j += (N_xx[1]/2)){
        efhatx(j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1])) = complex<double>(0.0,0.0);
        efhaty(j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1])) = complex<double>(0.0,0.0);
        efhatz(j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1])) = complex<double>(0.0,0.0);
      }
    }
    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());
    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatz.begin(),efz.begin());

    gt::stop("electric_field");

    // Main of K step
    gt::start("K");

    coeff(lr_sol.V, lr_sol.V, we_v.begin(), C1v);
    coeff(lr_sol.V, lr_sol.V, we_w.begin(), C1w);
    coeff(lr_sol.V, lr_sol.V, we_u.begin(), C1u);

    fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)tmpVhat.begin());

    ptw_mult_row(tmpVhat,lambdav_n.begin(),dVhat_v);
    ptw_mult_row(tmpVhat,lambdaw_n.begin(),dVhat_w);
    ptw_mult_row(tmpVhat,lambdau_n.begin(),dVhat_u);

    fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_v.begin(),dV_v.begin());
    fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_w.begin(),dV_w.begin());
    fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_u.begin(),dV_u.begin());

    coeff(lr_sol.V, dV_v, h_vv[0]*h_vv[1]*h_vv[2], C2v);
    coeff(lr_sol.V, dV_w, h_vv[0]*h_vv[1]*h_vv[2], C2w);
    coeff(lr_sol.V, dV_u, h_vv[0]*h_vv[1]*h_vv[2], C2u);

    schur(C1v, Tv, dcv_r, lwork);
    schur(C1w, Tw, dcw_r, lwork);
    schur(C1u, Tu, dcu_r, lwork);

    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    Tu.to_cplx(Tuc);
    C2v.to_cplx(C2vc);
    C2w.to_cplx(C2wc);
    C2u.to_cplx(C2uc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution
      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

      matmul(Khat,Tvc,Mhat);

      for(int rr = 0; rr < r; rr++){
        for(Index k = 0; k < N_xx[2]; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
              Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
              Mhat(idx,rr) *= exp(-ts_split*lambdax*dcv_r(rr));
            }
          }
        }
      }

      matmul_transb(Mhat,Tvc,Khat);

      // Full step -- Exact solution
      matmul(Khat,Twc,Mhat);

      for(int rr = 0; rr < r; rr++){
        for(Index k = 0; k < N_xx[2]; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
              Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
              Mhat(idx,rr) *= exp(-ts_split*lambday*dcw_r(rr))*ncxx;
            }
          }
        }
      }

      matmul_transb(Mhat,Twc,Khat);

      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin()); // TODO: could use FFTW_PRESERVE_INPUT here
      // NOT POSSIBLE FOR multidimensional transforms, see documentation

      // Full step -- Exponential Euler
      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());
      matmul(Khat,Tuc,Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.X,efx.begin(),Kex);
        ptw_mult_row(lr_sol.X,efy.begin(),Key);
        ptw_mult_row(lr_sol.X,efz.begin(),Kez);

        fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
        fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());
        fftw_execute_dft_r2c(plans_xx[0],Kez.begin(),(fftw_complex*)Kezhat.begin());

        matmul_transb(Kexhat,C2vc,Khat);
        matmul_transb(Keyhat,C2wc,tmpXhat);
        Khat += tmpXhat;

        matmul_transb(Kezhat,C2uc,tmpXhat);
        Khat += tmpXhat;

        matmul(Khat,Tuc,tmpXhat);

        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                Mhat(idx,rr) *= exp(-ts_ee*lambdaz*dcu_r(rr))*ncxx;
                Mhat(idx,rr) += ts_ee*phi1(-ts_ee*lambdaz*dcu_r(rr))*tmpXhat(idx,rr)*ncxx;
              }
            }
          }
        }

        matmul_transb(Mhat,Tuc,Khat);

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());
      }
    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_xx);

    gt::stop("K");

    // S Step

    gt::start("S");

    for(Index j = 0; j < (dxx_mult); j++){
      we_x(j) = efx(j) * h_xx[0] * h_xx[1] * h_xx[2];
      we_y(j) = efy(j) * h_xx[0] * h_xx[1] * h_xx[2];
      we_z(j) = efz(j) * h_xx[0] * h_xx[1] * h_xx[2];
    }

    coeff(lr_sol.X, lr_sol.X, we_x.begin(), D1x);
    coeff(lr_sol.X, lr_sol.X, we_y.begin(), D1y);
    coeff(lr_sol.X, lr_sol.X, we_z.begin(), D1z);

    fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)tmpXhat.begin());

    ptw_mult_row(tmpXhat,lambdax_n.begin(),dXhat_x);
    ptw_mult_row(tmpXhat,lambday_n.begin(),dXhat_y);
    ptw_mult_row(tmpXhat,lambdaz_n.begin(),dXhat_z);

    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_x.begin(),dX_x.begin());
    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_y.begin(),dX_y.begin());
    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_z.begin(),dX_z.begin());

    coeff(lr_sol.X, dX_x, h_xx[0]*h_xx[1]*h_xx[2], D2x);
    coeff(lr_sol.X, dX_y, h_xx[0]*h_xx[1]*h_xx[2], D2y);
    coeff(lr_sol.X, dX_z, h_xx[0]*h_xx[1]*h_xx[2], D2z);

    // Explicit Euler
    for(Index jj = 0; jj< nsteps_split; jj++){
      matmul_transb(lr_sol.S,C1v,tmpS);
      matmul(D2x,tmpS,Tv);

      matmul_transb(lr_sol.S,C1w,tmpS);
      matmul(D2y,tmpS,Tw);

      matmul_transb(lr_sol.S,C1u,tmpS);
      matmul(D2z,tmpS,Tu);

      Tv += Tw;
      Tv += Tu;

      matmul_transb(lr_sol.S,C2v,tmpS);
      matmul(D1x,tmpS,Tw);

      Tv -= Tw;

      matmul_transb(lr_sol.S,C2w,tmpS);
      matmul(D1y,tmpS,Tw);

      Tv -= Tw;

      matmul_transb(lr_sol.S,C2u,tmpS);
      matmul(D1z,tmpS,Tw);

      Tv -= Tw;

      Tv *= ts_split;
      lr_sol.S += Tv;
    }

    gt::stop("S");

    // L step - here we reuse some old variable names
    
    gt::start("L");

    tmpV = lr_sol.V;

    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    schur(D1x, Tv, dcv_r, lwork);
    schur(D1y, Tw, dcw_r, lwork);
    schur(D1z, Tu, dcu_r, lwork);

    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    Tu.to_cplx(Tuc);
    D2x.to_cplx(C2vc);
    D2y.to_cplx(C2wc);
    D2z.to_cplx(C2uc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());
      matmul(Lhat,Tvc,Nhat);

      for(int rr = 0; rr < r; rr++){
        for(Index k = 0; k < N_vv[2]; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
              Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

              Nhat(idx,rr) *= exp(ts_split*lambdav*dcv_r(rr));
            }
          }
        }
      }

      matmul_transb(Nhat,Tvc,Lhat);

      matmul(Lhat,Twc,Nhat);

      for(int rr = 0; rr < r; rr++){
        for(Index k = 0; k < N_vv[2]; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j);
              Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

              Nhat(idx,rr) *= exp(ts_split*lambdaw*dcw_r(rr))*ncvv;
            }
          }
        }
      }
      matmul_transb(Nhat,Twc,Lhat);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      // Full step -- Exponential euler
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin()); // TODO: use FFTW_PRESERVE_INPUT
      // NOT POSSIBLE FOR multidimensional transforms, see documentation

      matmul(Lhat,Tuc,Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.V,v.begin(),Lv);
        ptw_mult_row(lr_sol.V,w.begin(),Lw);
        ptw_mult_row(lr_sol.V,u.begin(),Lu);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lu.begin(),(fftw_complex*)Luhat.begin());

        matmul_transb(Lvhat,C2vc,Lhat);
        matmul_transb(Lwhat,C2wc,tmpVhat);
        Lhat += tmpVhat;
        matmul_transb(Luhat,C2uc,tmpVhat);
        Lhat += tmpVhat;

        matmul(Lhat,Tuc,tmpVhat);

        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(ts_ee*lambdau*dcu_r(rr))*ncvv;
                Nhat(idx,rr) -= ts_ee*phi1(ts_ee*lambdau*dcu_r(rr))*tmpVhat(idx,rr)*ncvv;
              }
            }
          }
        }

        matmul_transb(Nhat,Tuc,Lhat);

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());
      }

    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_vv);

    transpose_inplace(lr_sol.S);

    gt::stop("L");

    // Electric energy
    gt::start("output");
    el_energy = 0.0;
    for(Index ii = 0; ii < (dxx_mult); ii++){
      el_energy += 0.5*(pow(efx(ii),2)+pow(efy(ii),2)+pow(efz(ii),2))*h_xx[0]*h_xx[1]*h_xx[2];
    }
    cout << "Electric energy: " << el_energy << endl;
    el_energyf << el_energy << endl;

    // Error Mass
    coeff_one(lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],int_x);
    coeff_one(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],int_v);

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

    coeff_one(lr_sol.V,we_u2.begin(),int_v2);

    int_v += int_v2;

    matvec(lr_sol.S,int_v,rho);

    energy = el_energy;
    for(int ii = 0; ii < r; ii++){
      energy += 0.5*(int_x(ii)*rho(ii));
    }

    err_energy = abs(energy0-energy);

    cout << "Error in energy: " << err_energy << endl;
    err_energyf << err_energy << endl;


    gt::stop("output");
  }
  gt::stop("time_loop");


  el_energyf.close();
  err_massf.close();
  err_energyf.close();

  cout << gt::sorted_output() << endl;

  return 0;
}
