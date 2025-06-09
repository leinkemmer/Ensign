#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/kernels.hpp>
#include <generic/timer.hpp>
#include <generic/fft.hpp>

#include <cxxopts.hpp>

using namespace Ensign;
using namespace Ensign::Matrix;

bool CPU;

lr2<double> integration_first_order(array<Index,3> N_xx,array<Index,3> N_vv, int r,double tstar, Index nsteps, int nsteps_split, int nsteps_ee, int nsteps_rk4, array<double,6> lim_xx, array<double,6> lim_vv, lr2<double> lr_sol, array<fftw_plan,2> plans_e, array<fftw_plan,2> plans_xx, array<fftw_plan,2> plans_vv, const blas_ops& blas){
  double tau = tstar/nsteps;

  double tau_split = tau / nsteps_split;
  double tau_ee = tau_split / nsteps_ee;
  double tau_rk4 = tau / nsteps_rk4;

  gt::start("Initialization CPU");

  array<double,3> h_xx, h_vv;
  int jj = 0;
  for(int ii = 0; ii < 3; ii++){
    h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
    h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
    jj+=2;
  }

  Index dxx_mult = N_xx[0]*N_xx[1]*N_xx[2];
  Index dxxh_mult = N_xx[2]*N_xx[1]*(N_xx[0]/2 + 1);

  Index dvv_mult = N_vv[0]*N_vv[1]*N_vv[2];
  Index dvvh_mult = N_vv[2]*N_vv[1]*(N_vv[0]/2 + 1);

  multi_array<double,1> v({dvv_mult});
  multi_array<double,1> w({dvv_mult});
  multi_array<double,1> u({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index k = 0; k < N_vv[2]; k++){
    for(Index j = 0; j < N_vv[1]; j++){
      for(Index i = 0; i < N_vv[0]; i++){
        Index idx = i+j*N_vv[0] + k*(N_vv[0]*N_vv[1]);
        v(idx) = lim_vv[0] + i*h_vv[0];
        w(idx) = lim_vv[2] + j*h_vv[1];
        u(idx) = lim_vv[4] + k*h_vv[2];
      }
    }
  }

  orthogonalize gs(&blas);

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

  // Some FFT stuff for X and V

  multi_array<complex<double>,1> lambdax_n({dxxh_mult});
  multi_array<complex<double>,1> lambday_n({dxxh_mult});
  multi_array<complex<double>,1> lambdaz_n({dxxh_mult});

  double ncxx = 1.0 / (dxx_mult);

  Index mult_j;
  Index mult_k;

  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index k = 0; k < N_xx[2]; k++){
    for(Index j = 0; j < N_xx[1]; j++){
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
        if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }

        Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

        lambdax_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i)*ncxx;
        lambday_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j)*ncxx;
        lambdaz_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k)*ncxx;
      }
    }
  }
  #else
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
  #endif

  multi_array<complex<double>,1> lambdav_n({dvvh_mult});
  multi_array<complex<double>,1> lambdaw_n({dvvh_mult});
  multi_array<complex<double>,1> lambdau_n({dvvh_mult});

  double ncvv = 1.0 / (dvv_mult);

  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index k = 0; k < N_vv[2]; k++){
    for(Index j = 0; j < N_vv[1]; j++){
      for(Index i = 0; i < (N_vv[0]/2+1); i++){
        if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
        if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
        Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

        lambdav_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i)*ncvv;
        lambdaw_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j)*ncvv;
        lambdau_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k)*ncvv;
      }
    }
  }
  #else
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
  #endif

  multi_array<complex<double>,2> Khat({dxxh_mult,r});

  multi_array<complex<double>,2> Lhat({dvvh_mult,r});

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

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
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

  multi_array<complex<double>,2> D2xc({r,r});
  multi_array<complex<double>,2> D2yc({r,r});
  multi_array<complex<double>,2> D2zc({r,r});

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
  multi_array<double,1> dd1x_r({r});
  multi_array<double,1> dd1y_r({r});
  multi_array<double,1> dd1z_r({r});

  multi_array<double,2> Tv({r,r});
  multi_array<double,2> Tw({r,r});
  multi_array<double,2> Tu({r,r});
  multi_array<double,2> Tx({r,r});
  multi_array<double,2> Ty({r,r});
  multi_array<double,2> Tz({r,r});

  multi_array<complex<double>,2> Mhat({dxxh_mult,r});
  multi_array<complex<double>,2> Nhat({dvvh_mult,r});
  multi_array<complex<double>,2> Tvc({r,r});
  multi_array<complex<double>,2> Twc({r,r});
  multi_array<complex<double>,2> Tuc({r,r});
  multi_array<complex<double>,2> Txc({r,r});
  multi_array<complex<double>,2> Tyc({r,r});
  multi_array<complex<double>,2> Tzc({r,r});

  diagonalization schur(Tv.shape()[0]); // dumb call to obtain optimal value to work

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
  multi_array<double,2> tmpS1({r,r});
  multi_array<double,2> tmpS2({r,r});
  multi_array<double,2> tmpS3({r,r});
  multi_array<double,2> tmpS4({r,r});

  multi_array<complex<double>,2> tmpXhat({dxxh_mult,r});
  multi_array<complex<double>,2> tmpVhat({dvvh_mult,r});

  // Quantities of interest
  multi_array<double,1> int_x({r});
  multi_array<double,1> int_v({r});
  multi_array<double,1> int_v2({r});

  double mass0 = 0.0;
  double energy0 = 0.0;

  double mass = 0.0;
  double energy = 0.0;
  double el_energy = 0.0;
  double err_mass = 0.0;
  double err_energy = 0.0;

  // Initialization
  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(h_xx[0]*h_xx[1]*h_xx[2], dxx_mult);
  std::function<double(double*,double*)> ip_vv = inner_product_from_const_weight(h_vv[0]*h_vv[1]*h_vv[2], dvv_mult);

  gt::stop("Initialization CPU");

  gt::start("Computation initial QOI CPU");

  // Initial mass
  integrate(lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],int_x,blas);
  integrate(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],int_v,blas);

  blas.matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    mass0 += (int_x(ii)*rho(ii));
  }

  // Initial energy
  integrate(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],rho,blas);
  rho *= -1.0;
  blas.matmul(lr_sol.X,lr_sol.S,tmpX);
  blas.matvec(tmpX,rho,ef);
  ef += 1.0;
  fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index k = 0; k < N_xx[2]; k++){
    for(Index j = 0; j < N_xx[1]; j++){
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
        if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }

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
  #else
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
  #endif

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
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
  #ifdef __OPENMP__
  #pragma omp parallel for reduction(+:energy0)
  #endif
  for(Index ii = 0; ii < (dxx_mult); ii++){
    energy0 += 0.5*(pow(efx(ii),2)+pow(efy(ii),2)+pow(efz(ii),2))*h_xx[0]*h_xx[1]*h_xx[2];
  }

  multi_array<double,1> we_v2({dvv_mult});
  multi_array<double,1> we_w2({dvv_mult});
  multi_array<double,1> we_u2({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < (dvv_mult); j++){
    we_v2(j) = pow(v(j),2) * h_vv[0] * h_vv[1] * h_vv[2];
    we_w2(j) = pow(w(j),2) * h_vv[0] * h_vv[1] * h_vv[2];
    we_u2(j) = pow(u(j),2) * h_vv[0] * h_vv[1] * h_vv[2];
  }

  integrate(lr_sol.V,we_v2,int_v,blas);
  integrate(lr_sol.V,we_w2,int_v2,blas);

  int_v += int_v2;

  integrate(lr_sol.V,we_u2,int_v2,blas);

  int_v += int_v2;

  blas.matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    energy0 += 0.5*(int_x(ii)*rho(ii));
  }
  gt::stop("Computation initial QOI CPU");

  ofstream el_energyf;
  ofstream err_massf;
  ofstream err_energyf;
  if(CPU) {
    string str = "el_energy_order1_";
    str += to_string(nsteps);
    str += "_3d.txt";
    el_energyf.open(str);

    str = "err_mass_order1_";
    str += to_string(nsteps);
    str += "_3d.txt";
    err_massf.open(str);

    str = "err_energy_order1_";
    str += to_string(nsteps);
    str += "_3d.txt";
    err_energyf.open(str);

    el_energyf.precision(16);
    err_massf.precision(16);
    err_energyf.precision(16);

    el_energyf << tstar << endl;
    el_energyf << tau << endl;
  }

  //// FOR GPU ///

  #ifdef __CUDACC__

  gt::start("Initialization GPU");

  lr2<double> d_lr_sol(r,{dxx_mult,dvv_mult},stloc::device);
  d_lr_sol.X = lr_sol.X;
  d_lr_sol.V = lr_sol.V;
  d_lr_sol.S = lr_sol.S;

  // For Electric field
  multi_array<double,1> d_rho({r},stloc::device);

  multi_array<double,1> d_ef({dxx_mult},stloc::device);
  multi_array<double,1> d_efx({dxx_mult},stloc::device);
  multi_array<double,1> d_efy({dxx_mult},stloc::device);
  multi_array<double,1> d_efz({dxx_mult},stloc::device);

  multi_array<cuDoubleComplex,1> d_efhat({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhatx({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhaty({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhatz({dxxh_mult},stloc::device);

 // gt::start("Init GPU - EF - plan");
  array<cufftHandle,2> plans_d_e = create_plans_3d(N_xx,1);

  //cudaDeviceSynchronize();
  //gt::stop("Init GPU - EF - plan");

  //gt::start("Initialization GPU - FFT");
  // FFT stuff
  double* d_lim_xx;
  cudaMalloc((void**)&d_lim_xx,6*sizeof(double));
  cudaMemcpy(d_lim_xx,lim_xx.begin(),6*sizeof(double),cudaMemcpyHostToDevice);

  double* d_lim_vv;
  cudaMalloc((void**)&d_lim_vv,6*sizeof(double));
  cudaMemcpy(d_lim_vv,lim_vv.begin(),6*sizeof(double),cudaMemcpyHostToDevice);

  multi_array<cuDoubleComplex,2> d_Khat({dxxh_mult,r},stloc::device);
  array<cufftHandle,2> d_plans_xx = create_plans_3d(N_xx, r);

  multi_array<cuDoubleComplex,2> d_Lhat({dvvh_mult,r},stloc::device);
  array<cufftHandle,2> d_plans_vv = create_plans_3d(N_vv, r);

  //cudaDeviceSynchronize();
  //gt::stop("Initialization GPU - FFT");

  // C coefficients
  multi_array<double,2> d_C1v({r,r},stloc::device);
  multi_array<double,2> d_C1w({r,r}, stloc::device);
  multi_array<double,2> d_C1u({r,r}, stloc::device);

  multi_array<double,2> d_C2v({r,r},stloc::device);
  multi_array<double,2> d_C2w({r,r},stloc::device);
  multi_array<double,2> d_C2u({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_C2vc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2wc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2uc({r,r},stloc::device);

  multi_array<double,1> d_we_v({dvv_mult},stloc::device);
  d_we_v = we_v;

  multi_array<double,1> d_we_w({dvv_mult},stloc::device);
  d_we_w = we_w;

  multi_array<double,1> d_we_u({dvv_mult},stloc::device);
  d_we_u = we_u;

  multi_array<double,2> d_dV_v({dvv_mult,r},stloc::device);
  multi_array<double,2> d_dV_w({dvv_mult,r},stloc::device);
  multi_array<double,2> d_dV_u({dvv_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_dVhat_v({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dVhat_w({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dVhat_u({dvvh_mult,r},stloc::device);

  // D coefficients

  multi_array<double,2> d_D1x({r,r}, stloc::device);
  multi_array<double,2> d_D1y({r,r}, stloc::device);
  multi_array<double,2> d_D1z({r,r}, stloc::device);

  multi_array<double,2> d_D2x({r,r},stloc::device);
  multi_array<double,2> d_D2y({r,r},stloc::device);
  multi_array<double,2> d_D2z({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_D2xc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_D2yc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_D2zc({r,r},stloc::device);

  multi_array<double,1> d_we_x({dxx_mult},stloc::device);
  multi_array<double,1> d_we_y({dxx_mult},stloc::device);
  multi_array<double,1> d_we_z({dxx_mult},stloc::device);

  multi_array<double,2> d_dX_x({dxx_mult,r},stloc::device);
  multi_array<double,2> d_dX_y({dxx_mult,r},stloc::device);
  multi_array<double,2> d_dX_z({dxx_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_dXhat_x({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dXhat_y({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dXhat_z({dxxh_mult,r},stloc::device);

  multi_array<double,1> d_v({dvv_mult},stloc::device);
  d_v = v;
  multi_array<double,1> d_w({dvv_mult},stloc::device);
  d_w = w;
  multi_array<double,1> d_u({dvv_mult},stloc::device);
  d_u = u;

  // Schur decomposition

  multi_array<double,2> C1v_gpu({r,r});
  multi_array<double,2> Tv_gpu({r,r});
  multi_array<double,1> dcv_r_gpu({r});
  multi_array<double,2> C1w_gpu({r,r});
  multi_array<double,2> Tw_gpu({r,r});
  multi_array<double,1> dcw_r_gpu({r});
  multi_array<double,2> C1u_gpu({r,r});
  multi_array<double,2> Tu_gpu({r,r});
  multi_array<double,1> dcu_r_gpu({r});

  multi_array<double,2> D1x_gpu({r,r});
  multi_array<double,2> Tx_gpu({r,r});
  multi_array<double,1> dd1x_r_gpu({r});
  multi_array<double,2> D1y_gpu({r,r});
  multi_array<double,2> Ty_gpu({r,r});
  multi_array<double,1> dd1y_r_gpu({r});
  multi_array<double,2> D1z_gpu({r,r});
  multi_array<double,2> Tz_gpu({r,r});
  multi_array<double,1> dd1z_r_gpu({r});

  multi_array<double,1> d_dcv_r({r},stloc::device);
  multi_array<double,2> d_Tv({r,r},stloc::device);
  multi_array<double,1> d_dcw_r({r},stloc::device);
  multi_array<double,2> d_Tw({r,r},stloc::device);
  multi_array<double,1> d_dcu_r({r},stloc::device);
  multi_array<double,2> d_Tu({r,r},stloc::device);

  multi_array<double,1> d_dd1x_r({r},stloc::device);
  multi_array<double,2> d_Tx({r,r},stloc::device);
  multi_array<double,1> d_dd1y_r({r},stloc::device);
  multi_array<double,2> d_Ty({r,r},stloc::device);
  multi_array<double,1> d_dd1z_r({r},stloc::device);
  multi_array<double,2> d_Tz({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Mhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Nhat({dvvh_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Tvc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Twc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Tuc({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Txc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Tyc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Tzc({r,r},stloc::device);

  // For K step

  multi_array<double,2> d_Kex({dxx_mult,r},stloc::device);
  multi_array<double,2> d_Key({dxx_mult,r},stloc::device);
  multi_array<double,2> d_Kez({dxx_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Kexhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Keyhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Kezhat({dxxh_mult,r},stloc::device);

  // For L step

  multi_array<double,2> d_Lv({dvv_mult,r},stloc::device);
  multi_array<double,2> d_Lw({dvv_mult,r},stloc::device);
  multi_array<double,2> d_Lu({dvv_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Lvhat({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Lwhat({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Luhat({dvvh_mult,r},stloc::device);

  // Temporary to perform multiplications

  multi_array<double,2> d_tmpX({dxx_mult,r},stloc::device);
  multi_array<double,2> d_tmpS({r,r}, stloc::device);
  multi_array<double,2> d_tmpV({dvv_mult,r}, stloc::device);
  multi_array<double,2> d_tmpS1({r,r},stloc::device);
  multi_array<double,2> d_tmpS2({r,r},stloc::device);
  multi_array<double,2> d_tmpS3({r,r},stloc::device);
  multi_array<double,2> d_tmpS4({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_tmpXhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_tmpVhat({dvvh_mult,r},stloc::device);


  // Quantities of interest

  double* d_el_energy_x;
  cudaMalloc((void**)&d_el_energy_x,sizeof(double));
  double* d_el_energy_y;
  cudaMalloc((void**)&d_el_energy_y,sizeof(double));
  double* d_el_energy_z;
  cudaMalloc((void**)&d_el_energy_z,sizeof(double));

  double d_el_energy_CPU;

  multi_array<double,1> d_int_x({r},stloc::device);
  multi_array<double,1> d_int_v({r},stloc::device);

  double* d_mass;
  cudaMalloc((void**)&d_mass,sizeof(double));
  double d_mass_CPU;
  double err_mass_CPU;

  multi_array<double,1> d_we_v2({dvv_mult},stloc::device);
  multi_array<double,1> d_we_w2({dvv_mult},stloc::device);
  multi_array<double,1> d_we_u2({dvv_mult},stloc::device);
  d_we_v2 = we_v2;
  d_we_w2 = we_w2;
  d_we_u2 = we_u2;

  multi_array<double,1> d_int_v2({r},stloc::device);
  multi_array<double,1> d_int_v3({r},stloc::device);

  double* d_energy;
  cudaMalloc((void**)&d_energy,sizeof(double));
  double d_energy_CPU;
  double err_energy_CPU;

  cudaDeviceSynchronize();
  gt::stop("Initialization GPU");

  ofstream el_energyGPUf;
  ofstream err_massGPUf;
  ofstream err_energyGPUf;

  string strg = "el_energy_gpu_order1_";
  strg += to_string(nsteps);
  strg += "_3d.txt";
  el_energyGPUf.open(strg);

  strg = "err_mass_gpu_order1_";
  strg += to_string(nsteps);
  strg += "_3d.txt";
  err_massGPUf.open(strg);

  strg = "err_energy_gpu_order1_";
  strg += to_string(nsteps);
  strg += "_3d.txt";
  err_energyGPUf.open(strg);

  el_energyGPUf.precision(16);
  err_massGPUf.precision(16);
  err_energyGPUf.precision(16);

  el_energyGPUf << tstar << endl;
  el_energyGPUf << tau << endl;

  #endif

  ////////////////////
  //nsteps = 20;
  ////////////////////

  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i + 1 << " on " << nsteps << endl;

    // CPU

    if(CPU) {
      gt::start("Main loop CPU");

      /* K step */

      tmpX = lr_sol.X;
      blas.matmul(tmpX,lr_sol.S,lr_sol.X);

      gt::start("Electric Field CPU");

      // Electric field

      integrate(lr_sol.V,-h_vv[0]*h_vv[1]*h_vv[2],rho,blas);
      blas.matvec(lr_sol.X,rho,ef);
      ef += 1.0;
      fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

      //gt::start("Electric Field CPU - ptw");
      #ifdef __OPENMP__
      #pragma omp parallel for
      for(Index k = 0; k < N_xx[2]; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2+1); i++){
            if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
            if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
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
      #else
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
      #endif

      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index k = 0; k < (N_xx[2]/2 + 1); k += (N_xx[2]/2)){
        for(Index j = 0; j < (N_xx[1]/2 + 1); j += (N_xx[1]/2)){
          efhatx(j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1])) = complex<double>(0.0,0.0);
          efhaty(j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1])) = complex<double>(0.0,0.0);
          efhatz(j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1])) = complex<double>(0.0,0.0);
        }
      }
      //gt::stop("Electric Field CPU - ptw");

      fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
      fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());
      fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatz.begin(),efz.begin());

      gt::stop("Electric Field CPU");

      // Main of K step

      gt::start("C coeff CPU");
      coeff(lr_sol.V, lr_sol.V, we_v, C1v, blas);
      coeff(lr_sol.V, lr_sol.V, we_w, C1w, blas);
      coeff(lr_sol.V, lr_sol.V, we_u, C1u, blas);

      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)tmpVhat.begin());

      ptw_mult_row(tmpVhat,lambdav_n,dVhat_v);
      ptw_mult_row(tmpVhat,lambdaw_n,dVhat_w);
      ptw_mult_row(tmpVhat,lambdau_n,dVhat_u);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_v.begin(),dV_v.begin());
      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_w.begin(),dV_w.begin());
      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_u.begin(),dV_u.begin());

      coeff(lr_sol.V, dV_v, h_vv[0]*h_vv[1]*h_vv[2], C2v, blas);
      coeff(lr_sol.V, dV_w, h_vv[0]*h_vv[1]*h_vv[2], C2w, blas);
      coeff(lr_sol.V, dV_u, h_vv[0]*h_vv[1]*h_vv[2], C2u, blas);
      gt::stop("C coeff CPU");

      gt::start("Schur C CPU");
      schur(C1v, Tv, dcv_r);
      schur(C1w, Tw, dcw_r);
      schur(C1u, Tu, dcu_r);

      Tv.to_cplx(Tvc);
      Tw.to_cplx(Twc);
      Tu.to_cplx(Tuc);
      C2v.to_cplx(C2vc);
      C2w.to_cplx(C2wc);
      C2u.to_cplx(C2uc);

      gt::stop("Schur C CPU");

      gt::start("K step CPU");

      // Internal splitting
      for(Index ii = 0; ii < nsteps_split; ii++){
        // Full step -- Exact solution

        //gt::start("First split K CPU");
        fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

        blas.matmul(Khat,Tvc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split*lambdax*dcv_r(rr));
              }
            }
          }
        }


        blas.matmul_transb(Mhat,Tvc,Khat);

        //gt::stop("First split K CPU");
        // Full step -- Exact solution

        //gt::start("Second split K CPU");
        blas.matmul(Khat,Twc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split*lambday*dcw_r(rr))*ncxx;
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split*lambday*dcw_r(rr))*ncxx;
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Mhat,Twc,Khat);

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

        //gt::stop("Second split K CPU");

        // Full step --

        //gt::start("Third split K CPU");
        fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

        blas.matmul(Khat,Tuc,Mhat);

        for(Index jj = 0; jj < nsteps_ee; jj++){

          //gt::start("First stage Third split K CPU");
          ptw_mult_row(lr_sol.X,efx,Kex);
          ptw_mult_row(lr_sol.X,efy,Key);
          ptw_mult_row(lr_sol.X,efz,Kez);

          fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Kez.begin(),(fftw_complex*)Kezhat.begin());

          blas.matmul_transb(Kexhat,C2vc,Khat);
          blas.matmul_transb(Keyhat,C2wc,tmpXhat);

          Khat += tmpXhat;

          blas.matmul_transb(Kezhat,C2uc,tmpXhat);

          Khat += tmpXhat;

          blas.matmul(Khat,Tuc,tmpXhat);

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }

                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) *= exp(-tau_ee*lambdaz*dcu_r(rr));
                  Mhat(idx,rr) += tau_ee*phi1_im(-tau_ee*lambdaz*dcu_r(rr))*tmpXhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) *= exp(-tau_ee*lambdaz*dcu_r(rr));
                  Mhat(idx,rr) += tau_ee*phi1_im(-tau_ee*lambdaz*dcu_r(rr))*tmpXhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Mhat,Tuc,Khat);

          Khat *= ncxx;

          fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

          //gt::stop("First stage Third split K CPU");

          // Second stage

          //gt::start("Second stage Third split K CPU");

          ptw_mult_row(lr_sol.X,efx,Kex);
          ptw_mult_row(lr_sol.X,efy,Key);
          ptw_mult_row(lr_sol.X,efz,Kez);

          fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Kez.begin(),(fftw_complex*)Kezhat.begin());

          blas.matmul_transb(Kexhat,C2vc,Khat);
          blas.matmul_transb(Keyhat,C2wc,Kexhat);

          Kexhat += Khat;

          blas.matmul_transb(Kezhat,C2uc,Khat);

          Kexhat += Khat;

          blas.matmul(Kexhat,Tuc,Khat);

          Khat -= tmpXhat;

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }

                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) += tau_ee*phi2_im(-tau_ee*lambdaz*dcu_r(rr))*Khat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) += tau_ee*phi2_im(-tau_ee*lambdaz*dcu_r(rr))*Khat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Mhat,Tuc,Khat);

          Khat *= ncxx;

          fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

          //gt::stop("Second stage Third split K CPU");
        }

        //gt::stop("Third split K CPU");
      }

      gt::stop("K step CPU");

      gt::start("Gram Schmidt K CPU");
      gs(lr_sol.X, lr_sol.S, ip_xx);
      gt::stop("Gram Schmidt K CPU");


      /* S Step */

      gt::start("D coeff CPU");
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index j = 0; j < (dxx_mult); j++){
        we_x(j) = efx(j) * h_xx[0] * h_xx[1] * h_xx[2];
        we_y(j) = efy(j) * h_xx[0] * h_xx[1] * h_xx[2];
        we_z(j) = efz(j) * h_xx[0] * h_xx[1] * h_xx[2];
      }


      coeff(lr_sol.X, lr_sol.X, we_x, D1x, blas);
      coeff(lr_sol.X, lr_sol.X, we_y, D1y, blas);
      coeff(lr_sol.X, lr_sol.X, we_z, D1z, blas);


      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)tmpXhat.begin());

      ptw_mult_row(tmpXhat,lambdax_n,dXhat_x);
      ptw_mult_row(tmpXhat,lambday_n,dXhat_y);
      ptw_mult_row(tmpXhat,lambdaz_n,dXhat_z);


      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_x.begin(),dX_x.begin());
      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_y.begin(),dX_y.begin());
      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_z.begin(),dX_z.begin());


      coeff(lr_sol.X, dX_x, h_xx[0]*h_xx[1]*h_xx[2], D2x, blas);
      coeff(lr_sol.X, dX_y, h_xx[0]*h_xx[1]*h_xx[2], D2y, blas);
      coeff(lr_sol.X, dX_z, h_xx[0]*h_xx[1]*h_xx[2], D2z, blas);
      gt::stop("D coeff CPU");

      gt::start("S step CPU");
      // Rk4
      for(Index jj = 0; jj< nsteps_rk4; jj++){
        blas.matmul_transb(lr_sol.S,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS1);
        blas.matmul_transb(lr_sol.S,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS1 += Tw;
        blas.matmul_transb(lr_sol.S,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS1 += Tw;
        blas.matmul_transb(lr_sol.S,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS1 -= Tw;
        blas.matmul_transb(lr_sol.S,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS1 -= Tw;
        blas.matmul_transb(lr_sol.S,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS1 -= Tw;

        Tv = tmpS1;
        Tv *= (tau_rk4/2);
        Tv += lr_sol.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS2);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS2 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS2 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS2 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS2 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS2 -= Tw;

        Tv = tmpS2;
        Tv *= (tau_rk4/2);
        Tv += lr_sol.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS3);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS3 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS3 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS3 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS3 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS3 -= Tw;

        Tv = tmpS3;
        Tv *= tau_rk4;
        Tv += lr_sol.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS4);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS4 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS4 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS4 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS4 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS4 -= Tw;

        tmpS2 *= 2.0;
        tmpS3 *= 2.0;

        tmpS1 += tmpS2;
        tmpS1 += tmpS3;
        tmpS1 += tmpS4;
        tmpS1 *= (tau_rk4/6.0);

        lr_sol.S += tmpS1;

      }
      gt::stop("S step CPU");

      /* L step */

      tmpV = lr_sol.V;

      blas.matmul_transb(tmpV,lr_sol.S,lr_sol.V);


      gt::start("Schur L CPU");
      schur(D1x, Tx, dd1x_r);
      schur(D1y, Ty, dd1y_r);
      schur(D1z, Tz, dd1z_r);

      Tx.to_cplx(Txc);
      Ty.to_cplx(Tyc);
      Tz.to_cplx(Tzc);
      D2x.to_cplx(D2xc);
      D2y.to_cplx(D2yc);
      D2z.to_cplx(D2zc);

      gt::stop("Schur L CPU");

      gt::start("L step CPU");

      // Internal splitting
      for(Index ii = 0; ii < nsteps_split; ii++){

      // gt::start("First split L CPU");
        // Full step -- Exact solution
        fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

        blas.matmul(Lhat,Txc,Nhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split*lambdav*dd1x_r(rr));
              }
            }
          }
        }


        blas.matmul_transb(Nhat,Txc,Lhat);

        //gt::stop("First split L CPU");

        //gt::start("Second split L CPU");
        blas.matmul(Lhat,Tyc,Nhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
                complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split*lambdaw*dd1y_r(rr))*ncvv;
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split*lambdaw*dd1y_r(rr))*ncvv;
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Nhat,Tyc,Lhat);

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

        //gt::stop("Second split L CPU");

        //gt::start("Third split L CPU");
        // Full step --
        fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

        blas.matmul(Lhat,Tzc,Nhat);

        for(Index jj = 0; jj < nsteps_ee; jj++){

          //gt::start("First stage Third split L CPU");
          ptw_mult_row(lr_sol.V,v,Lv);
          ptw_mult_row(lr_sol.V,w,Lw);
          ptw_mult_row(lr_sol.V,u,Lu);

          fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lu.begin(),(fftw_complex*)Luhat.begin());

          blas.matmul_transb(Lvhat,D2xc,Lhat);

          blas.matmul_transb(Lwhat,D2yc,tmpVhat);

          Lhat += tmpVhat;

          blas.matmul_transb(Luhat,D2zc,tmpVhat);

          Lhat += tmpVhat;

          blas.matmul(Lhat,Tzc,tmpVhat);

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }

                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) *= exp(tau_ee*lambdau*dd1z_r(rr));
                  Nhat(idx,rr) -= tau_ee*phi1_im(tau_ee*lambdau*dd1z_r(rr))*tmpVhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) *= exp(tau_ee*lambdau*dd1z_r(rr));
                  Nhat(idx,rr) -= tau_ee*phi1_im(tau_ee*lambdau*dd1z_r(rr))*tmpVhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Nhat,Tzc,Lhat);

          Lhat *= ncvv;

          fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

          //gt::stop("First stage Third split L CPU");

          // Second stage

          //gt::start("Second stage Third split L CPU");

          ptw_mult_row(lr_sol.V,v,Lv);
          ptw_mult_row(lr_sol.V,w,Lw);
          ptw_mult_row(lr_sol.V,u,Lu);

          fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lu.begin(),(fftw_complex*)Luhat.begin());

          blas.matmul_transb(Lvhat,D2xc,Lhat);
          blas.matmul_transb(Lwhat,D2yc,Lvhat);

          Lvhat += Lhat;

          blas.matmul_transb(Luhat,D2zc,Lhat);

          Lvhat += Lhat;

          blas.matmul(Lvhat,Tzc,Lhat);

          Lhat -= tmpVhat;

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }

                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) -= tau_ee*phi2_im(tau_ee*lambdau*dd1z_r(rr))*Lhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) -= tau_ee*phi2_im(tau_ee*lambdau*dd1z_r(rr))*Lhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Nhat,Tzc,Lhat);

          Lhat *= ncvv;

          fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

          //gt::stop("Second stage Third split L CPU");

        }

        //gt::stop("Third split L CPU");

      }

      gt::stop("L step CPU");
      
      gt::start("Gram Schmidt L CPU");
      gs(lr_sol.V, lr_sol.S, ip_vv);
      gt::stop("Gram Schmidt L CPU");

      //gt::start("Transpose S CPU");
      transpose_inplace(lr_sol.S);
      //gt::stop("Transpose S CPU");

      gt::stop("Main loop CPU");

      gt::start("QOI CPU");
      // Electric energy
      el_energy = 0.0;
      #ifdef __OPENMP__
      #pragma omp parallel for reduction(+:el_energy)
      #endif
      for(Index ii = 0; ii < (dxx_mult); ii++){
        el_energy += 0.5*(pow(efx(ii),2)+pow(efy(ii),2)+pow(efz(ii),2))*h_xx[0]*h_xx[1]*h_xx[2];
      }

      el_energyf << el_energy << endl;

      // Error Mass
      integrate(lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],int_x,blas);
      integrate(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],int_v,blas);

      blas.matvec(lr_sol.S,int_v,rho);

      mass = 0.0;
      for(int ii = 0; ii < r; ii++){
        mass += (int_x(ii)*rho(ii));
      }

      err_mass = abs(mass0-mass);

      err_massf << err_mass << endl;

      // Error in energy

      integrate(lr_sol.V,we_v2,int_v,blas);
      integrate(lr_sol.V,we_w2,int_v2,blas);

      int_v += int_v2;

      integrate(lr_sol.V,we_u2,int_v2,blas);

      int_v += int_v2;

      blas.matvec(lr_sol.S,int_v,rho);

      energy = el_energy;
      for(int ii = 0; ii < r; ii++){
        energy += 0.5*(int_x(ii)*rho(ii));
      }

      err_energy = abs(energy0-energy);

      err_energyf << err_energy << endl;

      gt::stop("QOI CPU");

    }

    // GPU

    #ifdef __CUDACC__

    // K step

    gt::start("Main loop GPU");

    d_tmpX = d_lr_sol.X;

    blas.matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    gt::start("Electric Field GPU");

    integrate(d_lr_sol.V,-h_vv[0]*h_vv[1]*h_vv[2],d_rho,blas);

    blas.matvec(d_lr_sol.X,d_rho,d_ef);

    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier_3d<<<(dxxh_mult+n_threads-1)/n_threads,n_threads>>>(dxxh_mult, N_xx[0]/2+1, N_xx[1], N_xx[2], d_efhat.begin(), d_lim_xx, ncxx, d_efhatx.begin(), d_efhaty.begin(), d_efhatz.begin());

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatx.begin(),d_efx.begin());
    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhaty.begin(),d_efy.begin());
    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatz.begin(),d_efz.begin());

    cudaDeviceSynchronize();
    gt::stop("Electric Field GPU");

    // Main of K step

    gt::start("C coeff GPU");
    coeff(d_lr_sol.V, d_lr_sol.V, d_we_v, d_C1v, blas);
    coeff(d_lr_sol.V, d_lr_sol.V, d_we_w, d_C1w, blas);
    coeff(d_lr_sol.V, d_lr_sol.V, d_we_u, d_C1u, blas);

    cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_tmpVhat.begin());

    ptw_mult_row_cplx_fourier_3d<<<(dvvh_mult*r+n_threads-1)/n_threads,n_threads>>>(dvvh_mult*r, N_vv[0]/2+1, N_vv[1], N_vv[2], d_tmpVhat.begin(), d_lim_vv, ncvv, d_dVhat_v.begin(), d_dVhat_w.begin(), d_dVhat_u.begin());

    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_v.begin(),d_dV_v.begin());
    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_w.begin(),d_dV_w.begin());
    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_u.begin(),d_dV_u.begin());

    coeff(d_lr_sol.V, d_dV_v, h_vv[0]*h_vv[1]*h_vv[2], d_C2v, blas);
    coeff(d_lr_sol.V, d_dV_w, h_vv[0]*h_vv[1]*h_vv[2], d_C2w, blas);
    coeff(d_lr_sol.V, d_dV_u, h_vv[0]*h_vv[1]*h_vv[2], d_C2u, blas);

    cudaDeviceSynchronize();
    gt::stop("C coeff GPU");

    gt::start("Schur K GPU");

    C1v_gpu = d_C1v;
    schur(C1v_gpu, Tv_gpu, dcv_r_gpu);
    d_Tv = Tv_gpu;
    d_dcv_r = dcv_r_gpu;

    C1w_gpu = d_C1w;
    schur(C1w_gpu, Tw_gpu, dcw_r_gpu);
    d_Tw = Tw_gpu;
    d_dcw_r = dcw_r_gpu;
    C1u_gpu = d_C1u;
    schur(C1u_gpu, Tu_gpu, dcu_r_gpu);
    d_Tu = Tu_gpu;
    d_dcu_r = dcu_r_gpu;

    cplx_conv<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(), d_Tv.begin(), d_Tvc.begin());
    cplx_conv<<<(d_Tw.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tw.num_elements(), d_Tw.begin(), d_Twc.begin());
    cplx_conv<<<(d_Tu.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tu.num_elements(), d_Tu.begin(), d_Tuc.begin());

    cplx_conv<<<(d_C2v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2v.num_elements(), d_C2v.begin(), d_C2vc.begin());
    cplx_conv<<<(d_C2w.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2w.num_elements(), d_C2w.begin(), d_C2wc.begin());
    cplx_conv<<<(d_C2u.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2u.num_elements(), d_C2u.begin(), d_C2uc.begin());

    cudaDeviceSynchronize();
    gt::stop("Schur K GPU");

    gt::start("K step GPU");

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution

      //gt::start("First split K GPU");
      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

      blas.matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_3d_a<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcv_r.begin(), tau_split, d_lim_xx);

      blas.matmul_transb(d_Mhat,d_Tvc,d_Khat);

      //cudaDeviceSynchronize();
     // gt::stop("First split K GPU");
      // Full step -- Exact solution

     // gt::start("Second split K GPU");
      blas.matmul(d_Khat,d_Twc,d_Mhat);

      exact_sol_exp_3d_b<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcw_r.begin(), tau_split, d_lim_xx, ncxx);

      blas.matmul_transb(d_Mhat,d_Twc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      //cudaDeviceSynchronize();
      //gt::stop("Second split K GPU");

      // Full step --

      //gt::start("Third split K GPU");

      //#ifdef __FFTW__ // working on the server
      //ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);
      //#else
      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());
      //#endif

      blas.matmul(d_Khat,d_Tuc,d_Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        //gt::start("First stage Third split K GPU");
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efz.begin(),d_Kez.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Kez.begin(),(cufftDoubleComplex*)d_Kezhat.begin());

        blas.matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        blas.matmul_transb(d_Keyhat,d_C2wc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        blas.matmul_transb(d_Kezhat,d_C2uc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        blas.matmul(d_Khat,d_Tuc,d_tmpXhat);

        exp_euler_fourier_3d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(),d_dcu_r.begin(),tau_ee, d_lim_xx, d_tmpXhat.begin());

        blas.matmul_transb(d_Mhat,d_Tuc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

        //cudaDeviceSynchronize();
        //gt::stop("First stage Third split K GPU");
        // Second stage

        //gt::start("Second stage Third split K GPU");
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efz.begin(),d_Kez.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Kez.begin(),(cufftDoubleComplex*)d_Kezhat.begin());

        blas.matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        blas.matmul_transb(d_Keyhat,d_C2wc,d_Kexhat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        blas.matmul_transb(d_Kezhat,d_C2uc,d_Khat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        blas.matmul(d_Kexhat,d_Tuc,d_Khat);

        second_ord_stage_fourier_3d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(),d_dcu_r.begin(),tau_ee, d_lim_xx, d_tmpXhat.begin(), d_Khat.begin());

        blas.matmul_transb(d_Mhat,d_Tuc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

        //cudaDeviceSynchronize();
        //gt::stop("Second stage Third split K GPU");
      }
      //cudaDeviceSynchronize();
      //gt::stop("Third split K GPU");
    }
    cudaDeviceSynchronize();
    gt::stop("K step GPU");

    gt::start("Gram Schmidt K GPU");
    gs(d_lr_sol.X, d_lr_sol.S, h_xx[0]*h_xx[1]*h_xx[2]);
    cudaDeviceSynchronize();
    gt::stop("Gram Schmidt K GPU");


    // S step

    gt::start("D coeff GPU");
    ptw_mult_scal<<<(d_efx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efx.num_elements(), d_efx.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_x.begin());
    ptw_mult_scal<<<(d_efy.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efy.num_elements(), d_efy.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_y.begin());
    ptw_mult_scal<<<(d_efz.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efz.num_elements(), d_efz.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_z.begin());

    coeff(d_lr_sol.X, d_lr_sol.X, d_we_x, d_D1x, blas);
    coeff(d_lr_sol.X, d_lr_sol.X, d_we_y, d_D1y, blas);
    coeff(d_lr_sol.X, d_lr_sol.X, d_we_z, d_D1z, blas);

    cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_tmpXhat.begin());

    ptw_mult_row_cplx_fourier_3d<<<(dxxh_mult*r+n_threads-1)/n_threads,n_threads>>>(dxxh_mult*r, N_xx[0]/2+1, N_xx[1], N_xx[2], d_tmpXhat.begin(), d_lim_xx, ncxx, d_dXhat_x.begin(), d_dXhat_y.begin(), d_dXhat_z.begin());

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_x.begin(),d_dX_x.begin());
    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_y.begin(),d_dX_y.begin());
    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_z.begin(),d_dX_z.begin());

    coeff(d_lr_sol.X, d_dX_x, h_xx[0]*h_xx[1]*h_xx[2], d_D2x, blas);
    coeff(d_lr_sol.X, d_dX_y, h_xx[0]*h_xx[1]*h_xx[2], d_D2y, blas);
    coeff(d_lr_sol.X, d_dX_z, h_xx[0]*h_xx[1]*h_xx[2], d_D2z, blas);

    cudaDeviceSynchronize();
    gt::stop("D coeff GPU");

    gt::start("S step GPU");
    // RK4
    for(Index jj = 0; jj< nsteps_split; jj++){
      blas.matmul_transb(d_lr_sol.S,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS1);
      blas.matmul_transb(d_lr_sol.S,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(), d_Tv.begin(), tau_rk4/2.0, d_lr_sol.S.begin(), d_tmpS1.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS2);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(), d_Tv.begin(), tau_rk4/2.0, d_lr_sol.S.begin(), d_tmpS2.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS3);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(), d_Tv.begin(), tau_rk4, d_lr_sol.S.begin(), d_tmpS3.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS4);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4_finalcomb<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(), d_lr_sol.S.begin(), tau_rk4, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_Tw.begin());

    }
    cudaDeviceSynchronize();
    gt::stop("S step GPU");
    // L step

    d_tmpV = d_lr_sol.V;

    blas.matmul_transb(d_tmpV,d_lr_sol.S,d_lr_sol.V);

    gt::start("Schur L GPU");

    D1x_gpu = d_D1x;
    schur(D1x_gpu, Tx_gpu, dd1x_r_gpu);
    d_Tx = Tx_gpu;
    d_dd1x_r = dd1x_r_gpu;

    D1y_gpu = d_D1y;
    schur(D1y_gpu, Ty_gpu, dd1y_r_gpu);
    d_Ty = Ty_gpu;
    d_dd1y_r = dd1y_r_gpu;

    D1z_gpu = d_D1z;
    schur(D1z_gpu, Tz_gpu, dd1z_r_gpu);
    d_Tz = Tz_gpu;
    d_dd1z_r = dd1z_r_gpu;

    cplx_conv<<<(d_Tx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tx.num_elements(), d_Tx.begin(), d_Txc.begin());
    cplx_conv<<<(d_Ty.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Ty.num_elements(), d_Ty.begin(), d_Tyc.begin());
    cplx_conv<<<(d_Tz.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tz.num_elements(), d_Tz.begin(), d_Tzc.begin());

    cplx_conv<<<(d_D2x.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2x.num_elements(), d_D2x.begin(), d_D2xc.begin());
    cplx_conv<<<(d_D2y.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2y.num_elements(), d_D2y.begin(), d_D2yc.begin());
    cplx_conv<<<(d_D2z.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2z.num_elements(), d_D2z.begin(), d_D2zc.begin());

    cudaDeviceSynchronize();
    gt::stop("Schur L GPU");

    gt::start("L step GPU");

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){

      //gt::start("First split L GPU");
      // Full step -- Exact solution
      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

      blas.matmul(d_Lhat,d_Txc,d_Nhat);

      exact_sol_exp_3d_a<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dd1x_r.begin(), -tau_split, d_lim_vv);

      blas.matmul_transb(d_Nhat,d_Txc,d_Lhat);

      //cudaDeviceSynchronize();
      //gt::stop("First split L GPU");


      //gt::start("Second split L GPU");
      blas.matmul(d_Lhat,d_Tyc,d_Nhat);

      exact_sol_exp_3d_b<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dd1y_r.begin(), -tau_split, d_lim_vv, ncvv);

      blas.matmul_transb(d_Nhat,d_Tyc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      //cudaDeviceSynchronize();
      //gt::stop("Second split L GPU");

      //gt::start("Third split L GPU");

      // Full step --

      //#ifdef __FFTW__ // working on the server
      //ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), 1.0/ncvv);
      //#else
      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());
      //#endif

      blas.matmul(d_Lhat,d_Tzc,d_Nhat);


      for(Index jj = 0; jj < nsteps_ee; jj++){

        //gt::start("First stage Third split L GPU");

        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_u.begin(),d_Lu.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lu.begin(),(cufftDoubleComplex*)d_Luhat.begin());

        blas.matmul_transb(d_Lvhat,d_D2xc,d_Lhat);

        blas.matmul_transb(d_Lwhat,d_D2yc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        blas.matmul_transb(d_Luhat,d_D2zc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        blas.matmul(d_Lhat,d_Tzc,d_tmpVhat);

        exp_euler_fourier_3d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(),d_dd1z_r.begin(),-tau_ee, d_lim_vv, d_tmpVhat.begin());

        blas.matmul_transb(d_Nhat,d_Tzc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

        //cudaDeviceSynchronize();
       // gt::stop("First stage Third split L GPU");

        // Second stage

        //gt::start("Second stage Third split L GPU");
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_u.begin(),d_Lu.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lu.begin(),(cufftDoubleComplex*)d_Luhat.begin());

        blas.matmul_transb(d_Lvhat,d_D2xc,d_Lhat);
        blas.matmul_transb(d_Lwhat,d_D2yc,d_Lvhat);

        ptw_sum_complex<<<(d_Lvhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lvhat.num_elements(), d_Lvhat.begin(), d_Lhat.begin());

        blas.matmul_transb(d_Luhat,d_D2zc,d_Lhat);

        ptw_sum_complex<<<(d_Lvhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lvhat.num_elements(), d_Lvhat.begin(), d_Lhat.begin());

        blas.matmul(d_Lvhat,d_Tzc,d_Lhat);

        second_ord_stage_fourier_3d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(),d_dd1z_r.begin(),-tau_ee, d_lim_vv, d_tmpVhat.begin(), d_Lhat.begin());

        blas.matmul_transb(d_Nhat,d_Tzc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

        //cudaDeviceSynchronize();
        //gt::stop("Second stage Third split L GPU");
      }
      //cudaDeviceSynchronize();
      //gt::stop("Third split L GPU");
    }

    cudaDeviceSynchronize();
    gt::stop("L step GPU");
    
    gt::start("Gram Schmidt L GPU");
    gs(d_lr_sol.V, d_lr_sol.S, h_vv[0]*h_vv[1]*h_vv[2]);
    cudaDeviceSynchronize();
    gt::stop("Gram Schmidt L GPU");

    //gt::start("Transpose S GPU");
    //transpose_inplace<<<d_lr_sol.S.num_elements(),1>>>(r,d_lr_sol.S.begin());
    transpose_inplace(d_lr_sol.S);
    //cudaDeviceSynchronize();
    //gt::stop("Transpose S GPU");
    cudaDeviceSynchronize();
    gt::stop("Main loop GPU");

    gt::start("QOI GPU");

    // Electric energy

    cublasDdot (blas.handle_devres, d_efx.num_elements(), d_efx.begin(), 1, d_efx.begin(), 1, d_el_energy_x);
    cublasDdot (blas.handle_devres, d_efy.num_elements(), d_efy.begin(), 1, d_efy.begin(), 1, d_el_energy_y);
    cublasDdot (blas.handle_devres, d_efz.num_elements(), d_efz.begin(), 1, d_efz.begin(), 1, d_el_energy_z);
    cudaDeviceSynchronize();
    ptw_sum<<<1,1>>>(1,d_el_energy_x,d_el_energy_y);
    ptw_sum<<<1,1>>>(1,d_el_energy_x,d_el_energy_z);

    scale_unique<<<1,1>>>(d_el_energy_x,0.5*h_xx[0]*h_xx[1]*h_xx[2]);

    cudaMemcpy(&d_el_energy_CPU,d_el_energy_x,sizeof(double),cudaMemcpyDeviceToHost);

    el_energyGPUf << d_el_energy_CPU << endl;

    // Error mass

    integrate(d_lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],d_int_x,blas);
    integrate(d_lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],d_int_v,blas);

    blas.matvec(d_lr_sol.S,d_int_v,d_rho);

    cublasDdot (blas.handle_devres, r, d_int_x.begin(), 1, d_rho.begin(), 1,d_mass);
    cudaDeviceSynchronize();

    cudaMemcpy(&d_mass_CPU,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    err_mass_CPU = abs(mass0-d_mass_CPU);

    err_massGPUf << err_mass_CPU << endl;

    // Error energy

    integrate(d_lr_sol.V,d_we_v2,d_int_v,blas);
    integrate(d_lr_sol.V,d_we_w2,d_int_v2,blas);
    integrate(d_lr_sol.V,d_we_u2,d_int_v3,blas);

    ptw_sum_3mat<<<(d_int_v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_int_v.num_elements(),d_int_v.begin(),d_int_v2.begin(),d_int_v3.begin());

    blas.matvec(d_lr_sol.S,d_int_v,d_rho);

    cublasDdot (blas.handle_devres, r, d_int_x.begin(), 1, d_rho.begin(), 1, d_energy);
    cudaDeviceSynchronize();
    scale_unique<<<1,1>>>(d_energy,0.5);
    cudaMemcpy(&d_energy_CPU,d_energy,sizeof(double),cudaMemcpyDeviceToHost);

    err_energy_CPU = abs(energy0-(d_energy_CPU+d_el_energy_CPU));

    err_energyGPUf << err_energy_CPU << endl;

    cudaDeviceSynchronize();
    gt::stop("QOI GPU");

    #endif
  }

  if(CPU) {
    cout << "Electric energy: " << el_energy << endl;
  } else {
    #ifdef __CUDACC__
    cout << "Electric energy GPU: " << d_el_energy_CPU << endl;
    #endif
  }

  if(CPU) {
    cout << "Error in mass: " << err_mass << endl;
  } else {
    #ifdef __CUDACC__
    cout << "Error in mass GPU: " << err_mass_CPU << endl;
    #endif
  }

  if(CPU) {
    cout << "Error in energy: " << err_energy << endl;
  } else {
    #ifdef __CUDACC__
    cout << "Error in energy GPU: " << err_energy_CPU << endl;
    #endif
  }

  if(CPU) {
    el_energyf.close();
    err_massf.close();
    err_energyf.close();
  } else {
    #ifdef __CUDACC__
    destroy_plans(plans_d_e);
    destroy_plans(d_plans_xx);
    destroy_plans(d_plans_vv);

    el_energyGPUf.close();
    err_massGPUf.close();
    err_energyGPUf.close();
    #endif
  }

  #ifdef __CUDACC__
  lr_sol.X = d_lr_sol.X;
  lr_sol.S = d_lr_sol.S;
  lr_sol.V = d_lr_sol.V;

  cudaDeviceSynchronize();
  #endif

  return lr_sol;
}

lr2<double> integration_second_order(array<Index,3> N_xx,array<Index,3> N_vv, int r,double tstar, Index nsteps, int nsteps_split, int nsteps_ee, int nsteps_rk4, array<double,6> lim_xx, array<double,6> lim_vv, lr2<double> lr_sol, array<fftw_plan,2> plans_e, array<fftw_plan,2> plans_xx, array<fftw_plan,2> plans_vv, const blas_ops& blas){

  double tau = tstar/nsteps;
  double tau_h = tau/2.0;

  double tau_split = tau/nsteps_split;
  double tau_split_h = tau_h/nsteps_split;

  double tau_ee = tau_split / nsteps_ee;
  double tau_ee_h = tau_split_h / nsteps_ee;

  double tau_rk4_h = tau_h / nsteps_rk4;


  array<double,3> h_xx, h_vv;
  int jj = 0;
  for(int ii = 0; ii < 3; ii++){
    h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
    h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
    jj+=2;
  }

  Index dxx_mult = N_xx[0]*N_xx[1]*N_xx[2];
  Index dxxh_mult = N_xx[2]*N_xx[1]*(N_xx[0]/2 + 1);

  Index dvv_mult = N_vv[0]*N_vv[1]*N_vv[2];
  Index dvvh_mult = N_vv[2]*N_vv[1]*(N_vv[0]/2 + 1);

  multi_array<double,1> v({dvv_mult});
  multi_array<double,1> w({dvv_mult});
  multi_array<double,1> u({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index k = 0; k < N_vv[2]; k++){
    for(Index j = 0; j < N_vv[1]; j++){
      for(Index i = 0; i < N_vv[0]; i++){
        Index idx = i+j*N_vv[0] + k*(N_vv[0]*N_vv[1]);
        v(idx) = lim_vv[0] + i*h_vv[0];
        w(idx) = lim_vv[2] + j*h_vv[1];
        u(idx) = lim_vv[4] + k*h_vv[2];
      }
    }
  }

  orthogonalize gs(&blas);

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

  // Some FFT stuff for X and V

  multi_array<complex<double>,1> lambdax_n({dxxh_mult});
  multi_array<complex<double>,1> lambday_n({dxxh_mult});
  multi_array<complex<double>,1> lambdaz_n({dxxh_mult});

  double ncxx = 1.0 / (dxx_mult);

  Index mult_j;
  Index mult_k;

  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index k = 0; k < N_xx[2]; k++){
    for(Index j = 0; j < N_xx[1]; j++){
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
        if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }

        Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

        lambdax_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i)*ncxx;
        lambday_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j)*ncxx;
        lambdaz_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k)*ncxx;
      }
    }
  }
  #else
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
  #endif

  multi_array<complex<double>,1> lambdav_n({dvvh_mult});
  multi_array<complex<double>,1> lambdaw_n({dvvh_mult});
  multi_array<complex<double>,1> lambdau_n({dvvh_mult});

  double ncvv = 1.0 / (dvv_mult);

  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index k = 0; k < N_vv[2]; k++){
    for(Index j = 0; j < N_vv[1]; j++){
      for(Index i = 0; i < (N_vv[0]/2+1); i++){
        if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
        if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
        Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

        lambdav_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i)*ncvv;
        lambdaw_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j)*ncvv;
        lambdau_n(idx) = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k)*ncvv;
      }
    }
  }
  #else
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
  #endif

  multi_array<complex<double>,2> Khat({dxxh_mult,r});

  multi_array<complex<double>,2> Lhat({dvvh_mult,r});

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

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
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

  multi_array<complex<double>,2> D2xc({r,r});
  multi_array<complex<double>,2> D2yc({r,r});
  multi_array<complex<double>,2> D2zc({r,r});

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
  multi_array<double,1> dd1x_r({r});
  multi_array<double,1> dd1y_r({r});
  multi_array<double,1> dd1z_r({r});

  multi_array<double,2> Tv({r,r});
  multi_array<double,2> Tw({r,r});
  multi_array<double,2> Tu({r,r});
  multi_array<double,2> Tx({r,r});
  multi_array<double,2> Ty({r,r});
  multi_array<double,2> Tz({r,r});

  multi_array<complex<double>,2> Mhat({dxxh_mult,r});
  multi_array<complex<double>,2> Nhat({dvvh_mult,r});
  multi_array<complex<double>,2> Tvc({r,r});
  multi_array<complex<double>,2> Twc({r,r});
  multi_array<complex<double>,2> Tuc({r,r});
  multi_array<complex<double>,2> Txc({r,r});
  multi_array<complex<double>,2> Tyc({r,r});
  multi_array<complex<double>,2> Tzc({r,r});

  diagonalization schur(Tv.shape()[0]); // dumb call to obtain optimal value to work

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
  multi_array<double,2> tmpS1({r,r});
  multi_array<double,2> tmpS2({r,r});
  multi_array<double,2> tmpS3({r,r});
  multi_array<double,2> tmpS4({r,r});

  multi_array<complex<double>,2> tmpXhat({dxxh_mult,r});
  multi_array<complex<double>,2> tmpVhat({dvvh_mult,r});

  // Quantities of interest
  multi_array<double,1> int_x({r});
  multi_array<double,1> int_v({r});
  multi_array<double,1> int_v2({r});

  double mass0 = 0.0;
  double energy0 = 0.0;

  double mass = 0.0;
  double energy = 0.0;
  double el_energy = 0.0;
  double err_mass = 0.0;
  double err_energy = 0.0;

  // Initialization
  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(h_xx[0]*h_xx[1]*h_xx[2], dxx_mult);
  std::function<double(double*,double*)> ip_vv = inner_product_from_const_weight(h_vv[0]*h_vv[1]*h_vv[2], dvv_mult);

  // Initial mass
  integrate(lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],int_x,blas);
  integrate(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],int_v,blas);

  blas.matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    mass0 += (int_x(ii)*rho(ii));
  }

  // Initial energy
  integrate(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],rho,blas);
  rho *= -1.0;
  blas.matmul(lr_sol.X,lr_sol.S,tmpX);
  blas.matvec(tmpX,rho,ef);
  ef += 1.0;
  fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index k = 0; k < N_xx[2]; k++){
    for(Index j = 0; j < N_xx[1]; j++){
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
        if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }

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
  #else
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
  #endif

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
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
  #ifdef __OPENMP__
  #pragma omp parallel for reduction(+:energy0)
  #endif
  for(Index ii = 0; ii < (dxx_mult); ii++){
    energy0 += 0.5*(pow(efx(ii),2)+pow(efy(ii),2)+pow(efz(ii),2))*h_xx[0]*h_xx[1]*h_xx[2];
  }

  multi_array<double,1> we_v2({dvv_mult});
  multi_array<double,1> we_w2({dvv_mult});
  multi_array<double,1> we_u2({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < (dvv_mult); j++){
    we_v2(j) = pow(v(j),2) * h_vv[0] * h_vv[1] * h_vv[2];
    we_w2(j) = pow(w(j),2) * h_vv[0] * h_vv[1] * h_vv[2];
    we_u2(j) = pow(u(j),2) * h_vv[0] * h_vv[1] * h_vv[2];
  }

  integrate(lr_sol.V,we_v2,int_v,blas);
  integrate(lr_sol.V,we_w2,int_v2,blas);

  int_v += int_v2;

  integrate(lr_sol.V,we_u2,int_v2,blas);

  int_v += int_v2;

  blas.matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    energy0 += 0.5*(int_x(ii)*rho(ii));
  }

  ofstream el_energyf;
  ofstream err_massf;
  ofstream err_energyf;
  if(CPU) {
    string str = "el_energy_order2_";
    str += to_string(nsteps);
    str += "_3d.txt";
    el_energyf.open(str);

    str = "err_mass_order2_";
    str += to_string(nsteps);
    str += "_3d.txt";
    err_massf.open(str);

    str = "err_energy_order2_";
    str += to_string(nsteps);
    str += "_3d.txt";
    err_energyf.open(str);

    el_energyf.precision(16);
    err_massf.precision(16);
    err_energyf.precision(16);

    el_energyf << tstar << endl;
    el_energyf << tau << endl;
  }

  // Additional stuff for second order
  lr2<double> lr_sol_e(r,{dxx_mult,dvv_mult});

  //// FOR GPU ///

  #ifdef __CUDACC__

  lr2<double> d_lr_sol(r,{dxx_mult,dvv_mult},stloc::device);
  d_lr_sol.X = lr_sol.X;
  d_lr_sol.V = lr_sol.V;
  d_lr_sol.S = lr_sol.S;

  // For Electric field
  multi_array<double,1> d_rho({r},stloc::device);

  multi_array<double,1> d_ef({dxx_mult},stloc::device);
  multi_array<double,1> d_efx({dxx_mult},stloc::device);
  multi_array<double,1> d_efy({dxx_mult},stloc::device);
  multi_array<double,1> d_efz({dxx_mult},stloc::device);

  multi_array<cuDoubleComplex,1> d_efhat({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhatx({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhaty({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhatz({dxxh_mult},stloc::device);

  array<cufftHandle,2> plans_d_e = create_plans_3d(N_xx,1);

  // FFT stuff
  double* d_lim_xx;
  cudaMalloc((void**)&d_lim_xx,6*sizeof(double));
  cudaMemcpy(d_lim_xx,lim_xx.begin(),6*sizeof(double),cudaMemcpyHostToDevice);

  double* d_lim_vv;
  cudaMalloc((void**)&d_lim_vv,6*sizeof(double));
  cudaMemcpy(d_lim_vv,lim_vv.begin(),6*sizeof(double),cudaMemcpyHostToDevice);

  multi_array<cuDoubleComplex,2> d_Khat({dxxh_mult,r},stloc::device);
  array<cufftHandle,2> d_plans_xx = create_plans_3d(N_xx, r);

  multi_array<cuDoubleComplex,2> d_Lhat({dvvh_mult,r},stloc::device);
  array<cufftHandle,2> d_plans_vv = create_plans_3d(N_vv, r);

  // C coefficients
  multi_array<double,2> d_C1v({r,r},stloc::device);
  multi_array<double,2> d_C1w({r,r}, stloc::device);
  multi_array<double,2> d_C1u({r,r}, stloc::device);

  multi_array<double,2> d_C2v({r,r},stloc::device);
  multi_array<double,2> d_C2w({r,r},stloc::device);
  multi_array<double,2> d_C2u({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_C2vc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2wc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2uc({r,r},stloc::device);

  multi_array<double,1> d_we_v({dvv_mult},stloc::device);
  d_we_v = we_v;

  multi_array<double,1> d_we_w({dvv_mult},stloc::device);
  d_we_w = we_w;

  multi_array<double,1> d_we_u({dvv_mult},stloc::device);
  d_we_u = we_u;

  multi_array<double,2> d_dV_v({dvv_mult,r},stloc::device);
  multi_array<double,2> d_dV_w({dvv_mult,r},stloc::device);
  multi_array<double,2> d_dV_u({dvv_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_dVhat_v({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dVhat_w({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dVhat_u({dvvh_mult,r},stloc::device);

  // D coefficients

  multi_array<double,2> d_D1x({r,r}, stloc::device);
  multi_array<double,2> d_D1y({r,r}, stloc::device);
  multi_array<double,2> d_D1z({r,r}, stloc::device);

  multi_array<double,2> d_D2x({r,r},stloc::device);
  multi_array<double,2> d_D2y({r,r},stloc::device);
  multi_array<double,2> d_D2z({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_D2xc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_D2yc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_D2zc({r,r},stloc::device);

  multi_array<double,1> d_we_x({dxx_mult},stloc::device);
  multi_array<double,1> d_we_y({dxx_mult},stloc::device);
  multi_array<double,1> d_we_z({dxx_mult},stloc::device);

  multi_array<double,2> d_dX_x({dxx_mult,r},stloc::device);
  multi_array<double,2> d_dX_y({dxx_mult,r},stloc::device);
  multi_array<double,2> d_dX_z({dxx_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_dXhat_x({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dXhat_y({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dXhat_z({dxxh_mult,r},stloc::device);

  multi_array<double,1> d_v({dvv_mult},stloc::device);
  d_v = v;
  multi_array<double,1> d_w({dvv_mult},stloc::device);
  d_w = w;
  multi_array<double,1> d_u({dvv_mult},stloc::device);
  d_u = u;

  // Schur decomposition

  multi_array<double,2> C1v_gpu({r,r});
  multi_array<double,2> Tv_gpu({r,r});
  multi_array<double,1> dcv_r_gpu({r});
  multi_array<double,2> C1w_gpu({r,r});
  multi_array<double,2> Tw_gpu({r,r});
  multi_array<double,1> dcw_r_gpu({r});
  multi_array<double,2> C1u_gpu({r,r});
  multi_array<double,2> Tu_gpu({r,r});
  multi_array<double,1> dcu_r_gpu({r});

  multi_array<double,2> D1x_gpu({r,r});
  multi_array<double,2> Tx_gpu({r,r});
  multi_array<double,1> dd1x_r_gpu({r});
  multi_array<double,2> D1y_gpu({r,r});
  multi_array<double,2> Ty_gpu({r,r});
  multi_array<double,1> dd1y_r_gpu({r});
  multi_array<double,2> D1z_gpu({r,r});
  multi_array<double,2> Tz_gpu({r,r});
  multi_array<double,1> dd1z_r_gpu({r});

  multi_array<double,1> d_dcv_r({r},stloc::device);
  multi_array<double,2> d_Tv({r,r},stloc::device);
  multi_array<double,1> d_dcw_r({r},stloc::device);
  multi_array<double,2> d_Tw({r,r},stloc::device);
  multi_array<double,1> d_dcu_r({r},stloc::device);
  multi_array<double,2> d_Tu({r,r},stloc::device);

  multi_array<double,1> d_dd1x_r({r},stloc::device);
  multi_array<double,2> d_Tx({r,r},stloc::device);
  multi_array<double,1> d_dd1y_r({r},stloc::device);
  multi_array<double,2> d_Ty({r,r},stloc::device);
  multi_array<double,1> d_dd1z_r({r},stloc::device);
  multi_array<double,2> d_Tz({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Mhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Nhat({dvvh_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Tvc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Twc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Tuc({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Txc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Tyc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Tzc({r,r},stloc::device);

  // For K step

  multi_array<double,2> d_Kex({dxx_mult,r},stloc::device);
  multi_array<double,2> d_Key({dxx_mult,r},stloc::device);
  multi_array<double,2> d_Kez({dxx_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Kexhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Keyhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Kezhat({dxxh_mult,r},stloc::device);

  // For L step

  multi_array<double,2> d_Lv({dvv_mult,r},stloc::device);
  multi_array<double,2> d_Lw({dvv_mult,r},stloc::device);
  multi_array<double,2> d_Lu({dvv_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Lvhat({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Lwhat({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Luhat({dvvh_mult,r},stloc::device);

  // Temporary to perform multiplications

  multi_array<double,2> d_tmpX({dxx_mult,r},stloc::device);
  multi_array<double,2> d_tmpS({r,r}, stloc::device);
  multi_array<double,2> d_tmpV({dvv_mult,r}, stloc::device);
  multi_array<double,2> d_tmpS1({r,r},stloc::device);
  multi_array<double,2> d_tmpS2({r,r},stloc::device);
  multi_array<double,2> d_tmpS3({r,r},stloc::device);
  multi_array<double,2> d_tmpS4({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_tmpXhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_tmpVhat({dvvh_mult,r},stloc::device);


  // Quantities of interest

  double* d_el_energy_x;
  cudaMalloc((void**)&d_el_energy_x,sizeof(double));
  double* d_el_energy_y;
  cudaMalloc((void**)&d_el_energy_y,sizeof(double));
  double* d_el_energy_z;
  cudaMalloc((void**)&d_el_energy_z,sizeof(double));

  double d_el_energy_CPU;

  multi_array<double,1> d_int_x({r},stloc::device);
  multi_array<double,1> d_int_v({r},stloc::device);

  double* d_mass;
  cudaMalloc((void**)&d_mass,sizeof(double));
  double d_mass_CPU;
  double err_mass_CPU;

  multi_array<double,1> d_we_v2({dvv_mult},stloc::device);
  multi_array<double,1> d_we_w2({dvv_mult},stloc::device);
  multi_array<double,1> d_we_u2({dvv_mult},stloc::device);
  d_we_v2 = we_v2;
  d_we_w2 = we_w2;
  d_we_u2 = we_u2;

  multi_array<double,1> d_int_v2({r},stloc::device);
  multi_array<double,1> d_int_v3({r},stloc::device);

  double* d_energy;
  cudaMalloc((void**)&d_energy,sizeof(double));
  double d_energy_CPU;
  double err_energy_CPU;

  ofstream el_energyGPUf;
  ofstream err_massGPUf;
  ofstream err_energyGPUf;

  //string strg = "../../plots/el_energy_gpu_order2_";
  //strg += to_string(nsteps);
  //strg += "_3d.txt";
  string strg = "el_energy_rk";
  strg += to_string(r);
  strg += ".txt";
  el_energyGPUf.open(strg);

  //strg = "../../plots/err_mass_gpu_order2_";
  //strg += to_string(nsteps);
  //strg += "_3d.txt";
  strg = "err_mass_rk";
  strg += to_string(r);
  strg += ".txt";
  err_massGPUf.open(strg);

  //strg = "../../plots/err_energy_gpu_order2_";
  //strg += to_string(nsteps);
  //strg += "_3d.txt";
  strg = "err_energy_rk";
  strg += to_string(r);
  strg += ".txt";
  err_energyGPUf.open(strg);

  el_energyGPUf.precision(16);
  err_massGPUf.precision(16);
  err_energyGPUf.precision(16);

  el_energyGPUf << tstar << endl;
  el_energyGPUf << tau << endl;

  // Additional stuff for second order
  lr2<double> d_lr_sol_e(r,{dxx_mult,dvv_mult}, stloc::device);

  #endif

  ///////////////
  //nsteps = 10;
  ///////////////


  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i + 1 << " on " << nsteps << endl;

    // CPU
    if(CPU) {
      gt::start("Total time CPU");

      /* Lie splitting to obtain the electric field */

      lr_sol_e.X = lr_sol.X;
      lr_sol_e.S = lr_sol.S;
      lr_sol_e.V = lr_sol.V;

      // Full step K until tau/2

      gt::start("Lie splitting for electric field CPU");

      tmpX = lr_sol_e.X;
      blas.matmul(tmpX,lr_sol_e.S,lr_sol_e.X);

      // Electric field

      integrate(lr_sol_e.V,-h_vv[0]*h_vv[1]*h_vv[2],rho,blas);

      blas.matvec(lr_sol_e.X,rho,ef);

      ef += 1.0;

      fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

      #ifdef __OPENMP__
      #pragma omp parallel for
      for(Index k = 0; k < N_xx[2]; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2+1); i++){
            if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
            if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
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
      #else
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
      #endif

      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
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

      // Electric energy
      el_energy = 0.0;
      #ifdef __OPENMP__
      #pragma omp parallel for reduction(+:el_energy)
      #endif
      for(Index ii = 0; ii < (dxx_mult); ii++){
        el_energy += 0.5*(pow(efx(ii),2)+pow(efy(ii),2)+pow(efz(ii),2))*h_xx[0]*h_xx[1]*h_xx[2];
      }

      el_energyf << el_energy << endl;

      // Main of K step

      coeff(lr_sol_e.V, lr_sol_e.V, we_v, C1v, blas);
      coeff(lr_sol_e.V, lr_sol_e.V, we_w, C1w, blas);
      coeff(lr_sol_e.V, lr_sol_e.V, we_u, C1u, blas);

      fftw_execute_dft_r2c(plans_vv[0],lr_sol_e.V.begin(),(fftw_complex*)tmpVhat.begin());

      ptw_mult_row(tmpVhat,lambdav_n,dVhat_v);
      ptw_mult_row(tmpVhat,lambdaw_n,dVhat_w);
      ptw_mult_row(tmpVhat,lambdau_n,dVhat_u);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_v.begin(),dV_v.begin());
      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_w.begin(),dV_w.begin());
      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_u.begin(),dV_u.begin());

      coeff(lr_sol_e.V, dV_v, h_vv[0]*h_vv[1]*h_vv[2], C2v, blas);
      coeff(lr_sol_e.V, dV_w, h_vv[0]*h_vv[1]*h_vv[2], C2w, blas);
      coeff(lr_sol_e.V, dV_u, h_vv[0]*h_vv[1]*h_vv[2], C2u, blas);

      schur(C1v, Tv, dcv_r);
      schur(C1w, Tw, dcw_r);
      schur(C1u, Tu, dcu_r);

      Tv.to_cplx(Tvc);
      Tw.to_cplx(Twc);
      Tu.to_cplx(Tuc);
      C2v.to_cplx(C2vc);
      C2w.to_cplx(C2wc);
      C2u.to_cplx(C2uc);

      // Internal splitting
      for(Index ii = 0; ii < nsteps_split; ii++){
        // Full step -- Exact solution

        fftw_execute_dft_r2c(plans_xx[0],lr_sol_e.X.begin(),(fftw_complex*)Khat.begin());

        blas.matmul(Khat,Tvc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h*lambdax*dcv_r(rr));
              }
            }
          }
        }

        blas.matmul_transb(Mhat,Tvc,Khat);

        // Full step -- Exact solution

        blas.matmul(Khat,Twc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h*lambday*dcw_r(rr))*ncxx;
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h*lambday*dcw_r(rr))*ncxx;
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Mhat,Twc,Khat);

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol_e.X.begin());

        // Full step --

        fftw_execute_dft_r2c(plans_xx[0],lr_sol_e.X.begin(),(fftw_complex*)Khat.begin());

        blas.matmul(Khat,Tuc,Mhat);

        for(Index jj = 0; jj < nsteps_ee; jj++){

          ptw_mult_row(lr_sol_e.X,efx,Kex);
          ptw_mult_row(lr_sol_e.X,efy,Key);
          ptw_mult_row(lr_sol_e.X,efz,Kez);

          fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Kez.begin(),(fftw_complex*)Kezhat.begin());

          blas.matmul_transb(Kexhat,C2vc,Khat);
          blas.matmul_transb(Keyhat,C2wc,tmpXhat);

          Khat += tmpXhat;

          blas.matmul_transb(Kezhat,C2uc,tmpXhat);

          Khat += tmpXhat;

          blas.matmul(Khat,Tuc,tmpXhat);

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }

                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) *= exp(-tau_ee_h*lambdaz*dcu_r(rr));
                  Mhat(idx,rr) += tau_ee_h*phi1_im(-tau_ee_h*lambdaz*dcu_r(rr))*tmpXhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) *= exp(-tau_ee_h*lambdaz*dcu_r(rr));
                  Mhat(idx,rr) += tau_ee_h*phi1_im(-tau_ee_h*lambdaz*dcu_r(rr))*tmpXhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Mhat,Tuc,Khat);

          Khat *= ncxx;

          fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol_e.X.begin());

          // Second stage

          ptw_mult_row(lr_sol_e.X,efx,Kex);
          ptw_mult_row(lr_sol_e.X,efy,Key);
          ptw_mult_row(lr_sol_e.X,efz,Kez);

          fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Kez.begin(),(fftw_complex*)Kezhat.begin());

          blas.matmul_transb(Kexhat,C2vc,Khat);
          blas.matmul_transb(Keyhat,C2wc,Kexhat);

          Kexhat += Khat;

          blas.matmul_transb(Kezhat,C2uc,Khat);

          Kexhat += Khat;

          blas.matmul(Kexhat,Tuc,Khat);

          Khat -= tmpXhat;

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }

                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) += tau_ee_h*phi2_im(-tau_ee_h*lambdaz*dcu_r(rr))*Khat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) += tau_ee_h*phi2_im(-tau_ee_h*lambdaz*dcu_r(rr))*Khat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Mhat,Tuc,Khat);

          Khat *= ncxx;

          fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol_e.X.begin());

        }

      }

      gs(lr_sol_e.X, lr_sol_e.S, ip_xx);

      // Full step S until tau/2

      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index j = 0; j < (dxx_mult); j++){
        we_x(j) = efx(j) * h_xx[0] * h_xx[1] * h_xx[2];
        we_y(j) = efy(j) * h_xx[0] * h_xx[1] * h_xx[2];
        we_z(j) = efz(j) * h_xx[0] * h_xx[1] * h_xx[2];
      }

      coeff(lr_sol_e.X, lr_sol_e.X, we_x, D1x, blas);
      coeff(lr_sol_e.X, lr_sol_e.X, we_y, D1y, blas);
      coeff(lr_sol_e.X, lr_sol_e.X, we_z, D1z, blas);

      fftw_execute_dft_r2c(plans_xx[0],lr_sol_e.X.begin(),(fftw_complex*)tmpXhat.begin());

      ptw_mult_row(tmpXhat,lambdax_n,dXhat_x);
      ptw_mult_row(tmpXhat,lambday_n,dXhat_y);
      ptw_mult_row(tmpXhat,lambdaz_n,dXhat_z);


      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_x.begin(),dX_x.begin());
      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_y.begin(),dX_y.begin());
      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_z.begin(),dX_z.begin());


      coeff(lr_sol_e.X, dX_x, h_xx[0]*h_xx[1]*h_xx[2], D2x, blas);
      coeff(lr_sol_e.X, dX_y, h_xx[0]*h_xx[1]*h_xx[2], D2y, blas);
      coeff(lr_sol_e.X, dX_z, h_xx[0]*h_xx[1]*h_xx[2], D2z, blas);

      // RK4
      for(Index jj = 0; jj< nsteps_rk4; jj++){
        blas.matmul_transb(lr_sol_e.S,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS1);
        blas.matmul_transb(lr_sol_e.S,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS1 += Tw;
        blas.matmul_transb(lr_sol_e.S,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS1 += Tw;
        blas.matmul_transb(lr_sol_e.S,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS1 -= Tw;
        blas.matmul_transb(lr_sol_e.S,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS1 -= Tw;
        blas.matmul_transb(lr_sol_e.S,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS1 -= Tw;

        Tv = tmpS1;
        Tv *= (tau_rk4_h/2);
        Tv += lr_sol_e.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS2);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS2 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS2 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS2 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS2 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS2 -= Tw;

        Tv = tmpS2;
        Tv *= (tau_rk4_h/2);
        Tv += lr_sol_e.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS3);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS3 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS3 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS3 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS3 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS3 -= Tw;

        Tv = tmpS3;
        Tv *= tau_rk4_h;
        Tv += lr_sol_e.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS4);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS4 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS4 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS4 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS4 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS4 -= Tw;

        tmpS2 *= 2.0;
        tmpS3 *= 2.0;

        tmpS1 += tmpS2;
        tmpS1 += tmpS3;
        tmpS1 += tmpS4;
        tmpS1 *= (tau_rk4_h/6.0);

        lr_sol_e.S += tmpS1;

      }

      // Full step L until tau/2

      tmpV = lr_sol_e.V;

      blas.matmul_transb(tmpV,lr_sol_e.S,lr_sol_e.V);

      schur(D1x, Tx, dd1x_r);
      schur(D1y, Ty, dd1y_r);
      schur(D1z, Tz, dd1z_r);

      Tx.to_cplx(Txc);
      Ty.to_cplx(Tyc);
      Tz.to_cplx(Tzc);
      D2x.to_cplx(D2xc);
      D2y.to_cplx(D2yc);
      D2z.to_cplx(D2zc);

      // Internal splitting
      for(Index ii = 0; ii < nsteps_split; ii++){

        // Full step -- Exact solution
        fftw_execute_dft_r2c(plans_vv[0],lr_sol_e.V.begin(),(fftw_complex*)Lhat.begin());

        blas.matmul(Lhat,Txc,Nhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split_h*lambdav*dd1x_r(rr));
              }
            }
          }
        }


        blas.matmul_transb(Nhat,Txc,Lhat);

        blas.matmul(Lhat,Tyc,Nhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
                complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split_h*lambdaw*dd1y_r(rr))*ncvv;
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split_h*lambdaw*dd1y_r(rr))*ncvv;
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Nhat,Tyc,Lhat);

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol_e.V.begin());

        // Full step --
        fftw_execute_dft_r2c(plans_vv[0],lr_sol_e.V.begin(),(fftw_complex*)Lhat.begin());

        blas.matmul(Lhat,Tzc,Nhat);

        for(Index jj = 0; jj < nsteps_ee; jj++){

          ptw_mult_row(lr_sol_e.V,v,Lv);
          ptw_mult_row(lr_sol_e.V,w,Lw);
          ptw_mult_row(lr_sol_e.V,u,Lu);

          fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lu.begin(),(fftw_complex*)Luhat.begin());

          blas.matmul_transb(Lvhat,D2xc,Lhat);

          blas.matmul_transb(Lwhat,D2yc,tmpVhat);

          Lhat += tmpVhat;

          blas.matmul_transb(Luhat,D2zc,tmpVhat);

          Lhat += tmpVhat;

          blas.matmul(Lhat,Tzc,tmpVhat);

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }

                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) *= exp(tau_ee_h*lambdau*dd1z_r(rr));
                  Nhat(idx,rr) -= tau_ee_h*phi1_im(tau_ee_h*lambdau*dd1z_r(rr))*tmpVhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) *= exp(tau_ee_h*lambdau*dd1z_r(rr));
                  Nhat(idx,rr) -= tau_ee_h*phi1_im(tau_ee_h*lambdau*dd1z_r(rr))*tmpVhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Nhat,Tzc,Lhat);

          Lhat *= ncvv;

          fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol_e.V.begin());

          // Second stage

          ptw_mult_row(lr_sol_e.V,v,Lv);
          ptw_mult_row(lr_sol_e.V,w,Lw);
          ptw_mult_row(lr_sol_e.V,u,Lu);

          fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lu.begin(),(fftw_complex*)Luhat.begin());

          blas.matmul_transb(Lvhat,D2xc,Lhat);
          blas.matmul_transb(Lwhat,D2yc,Lvhat);

          Lvhat += Lhat;

          blas.matmul_transb(Luhat,D2zc,Lhat);

          Lvhat += Lhat;

          blas.matmul(Lvhat,Tzc,Lhat);

          Lhat -= tmpVhat;

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }

                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) -= tau_ee_h*phi2_im(tau_ee_h*lambdau*dd1z_r(rr))*Lhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) -= tau_ee_h*phi2_im(tau_ee_h*lambdau*dd1z_r(rr))*Lhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Nhat,Tzc,Lhat);

          Lhat *= ncvv;

          fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol_e.V.begin());

        }

      }

      gt::stop("Lie splitting for electric field CPU");

      gt::start("Restarted integration CPU");

      gt::start("Electric field CPU");

      // Electric field at time tau/2

      integrate(lr_sol_e.V,-h_vv[0]*h_vv[1]*h_vv[2],rho,blas);

      blas.matvec(lr_sol_e.X,rho,ef);

      ef += 1.0;

      fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

      #ifdef __OPENMP__
      #pragma omp parallel for
      for(Index k = 0; k < N_xx[2]; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2+1); i++){
            if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
            if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
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
      #else
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
      #endif

      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
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

      gt::stop("Electric field CPU");

      // Here I have the electric field at time tau/2, so restart integration

      // Half step K (until tau/2)

      tmpX = lr_sol.X;
      blas.matmul(tmpX,lr_sol.S,lr_sol.X);

      gt::start("First half step K step CPU");

      // Internal splitting
      for(Index ii = 0; ii < nsteps_split; ii++){
        // Half step -- Exact solution

        fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

        blas.matmul(Khat,Tvc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambdax*dcv_r(rr));
              }
            }
          }
        }

        blas.matmul_transb(Mhat,Tvc,Khat);

        // Half step -- Exact solution

        blas.matmul(Khat,Twc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambday*dcw_r(rr))*ncxx;
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambday*dcw_r(rr))*ncxx;
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Mhat,Twc,Khat);

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

        // Full step -- exponential integrator

        fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

        blas.matmul(Khat,Tuc,Mhat);

        for(Index jj = 0; jj < nsteps_ee; jj++){

          ptw_mult_row(lr_sol.X,efx,Kex);
          ptw_mult_row(lr_sol.X,efy,Key);
          ptw_mult_row(lr_sol.X,efz,Kez);

          fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Kez.begin(),(fftw_complex*)Kezhat.begin());

          blas.matmul_transb(Kexhat,C2vc,Khat);
          blas.matmul_transb(Keyhat,C2wc,tmpXhat);

          Khat += tmpXhat;

          blas.matmul_transb(Kezhat,C2uc,tmpXhat);

          Khat += tmpXhat;

          blas.matmul(Khat,Tuc,tmpXhat);

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }

                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) *= exp(-tau_ee_h*lambdaz*dcu_r(rr));
                  Mhat(idx,rr) += tau_ee_h*phi1_im(-tau_ee_h*lambdaz*dcu_r(rr))*tmpXhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) *= exp(-tau_ee_h*lambdaz*dcu_r(rr));
                  Mhat(idx,rr) += tau_ee_h*phi1_im(-tau_ee_h*lambdaz*dcu_r(rr))*tmpXhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Mhat,Tuc,Khat);

          Khat *= ncxx;

          fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

          // Second stage

          ptw_mult_row(lr_sol.X,efx,Kex);
          ptw_mult_row(lr_sol.X,efy,Key);
          ptw_mult_row(lr_sol.X,efz,Kez);

          fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Kez.begin(),(fftw_complex*)Kezhat.begin());

          blas.matmul_transb(Kexhat,C2vc,Khat);
          blas.matmul_transb(Keyhat,C2wc,Kexhat);

          Kexhat += Khat;

          blas.matmul_transb(Kezhat,C2uc,Khat);

          Kexhat += Khat;

          blas.matmul(Kexhat,Tuc,Khat);

          Khat -= tmpXhat;

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }

                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) += tau_ee_h*phi2_im(-tau_ee_h*lambdaz*dcu_r(rr))*Khat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) += tau_ee_h*phi2_im(-tau_ee_h*lambdaz*dcu_r(rr))*Khat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Mhat,Tuc,Khat);

          if(jj != (nsteps_ee -1)){
            Khat *= ncxx;
            fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());
          }
        }

        // Half step -- exact solution

        blas.matmul(Khat,Twc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambday*dcw_r(rr));
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambday*dcw_r(rr));
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Mhat,Twc,Khat);

        // Half step -- Exact solution

        blas.matmul(Khat,Tvc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambdax*dcv_r(rr))*ncxx;
              }
            }
          }
        }

        blas.matmul_transb(Mhat,Tvc,Khat);

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

      }


      gs(lr_sol.X, lr_sol.S, ip_xx);

      gt::stop("First half step K step CPU");

      // Half step S (until tau/2)

      gt::start("D coeff CPU");
      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(Index j = 0; j < (dxx_mult); j++){
        we_x(j) = efx(j) * h_xx[0] * h_xx[1] * h_xx[2];
        we_y(j) = efy(j) * h_xx[0] * h_xx[1] * h_xx[2];
        we_z(j) = efz(j) * h_xx[0] * h_xx[1] * h_xx[2];
      }


      coeff(lr_sol.X, lr_sol.X, we_x, D1x, blas);
      coeff(lr_sol.X, lr_sol.X, we_y, D1y, blas);
      coeff(lr_sol.X, lr_sol.X, we_z, D1z, blas);

      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)tmpXhat.begin());

      ptw_mult_row(tmpXhat,lambdax_n,dXhat_x);
      ptw_mult_row(tmpXhat,lambday_n,dXhat_y);
      ptw_mult_row(tmpXhat,lambdaz_n,dXhat_z);


      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_x.begin(),dX_x.begin());
      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_y.begin(),dX_y.begin());
      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_z.begin(),dX_z.begin());


      coeff(lr_sol.X, dX_x, h_xx[0]*h_xx[1]*h_xx[2], D2x, blas);
      coeff(lr_sol.X, dX_y, h_xx[0]*h_xx[1]*h_xx[2], D2y, blas);
      coeff(lr_sol.X, dX_z, h_xx[0]*h_xx[1]*h_xx[2], D2z, blas);

      gt::stop("D coeff CPU");

      gt::start("First half step S step CPU");

      // Rk4
      for(Index jj = 0; jj< nsteps_rk4; jj++){
        blas.matmul_transb(lr_sol.S,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS1);
        blas.matmul_transb(lr_sol.S,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS1 += Tw;
        blas.matmul_transb(lr_sol.S,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS1 += Tw;
        blas.matmul_transb(lr_sol.S,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS1 -= Tw;
        blas.matmul_transb(lr_sol.S,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS1 -= Tw;
        blas.matmul_transb(lr_sol.S,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS1 -= Tw;

        Tv = tmpS1;
        Tv *= (tau_rk4_h/2);
        Tv += lr_sol.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS2);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS2 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS2 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS2 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS2 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS2 -= Tw;

        Tv = tmpS2;
        Tv *= (tau_rk4_h/2);
        Tv += lr_sol.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS3);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS3 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS3 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS3 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS3 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS3 -= Tw;

        Tv = tmpS3;
        Tv *= tau_rk4_h;
        Tv += lr_sol.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS4);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS4 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS4 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS4 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS4 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS4 -= Tw;

        tmpS2 *= 2.0;
        tmpS3 *= 2.0;

        tmpS1 += tmpS2;
        tmpS1 += tmpS3;
        tmpS1 += tmpS4;
        tmpS1 *= (tau_rk4_h/6.0);

        lr_sol.S += tmpS1;

      }

      gt::stop("First half step S step CPU");

      tmpV = lr_sol.V;
      blas.matmul_transb(tmpV,lr_sol.S,lr_sol.V);

      schur(D1x, Tx, dd1x_r);
      schur(D1y, Ty, dd1y_r);
      schur(D1z, Tz, dd1z_r);

      Tx.to_cplx(Txc);
      Ty.to_cplx(Tyc);
      Tz.to_cplx(Tzc);
      D2x.to_cplx(D2xc);
      D2y.to_cplx(D2yc);
      D2z.to_cplx(D2zc);

      gt::start("Full step L step CPU");

      // Internal splitting
      for(Index ii = 0; ii < nsteps_split; ii++){

        // Half step -- Exact solution
        fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

        blas.matmul(Lhat,Txc,Nhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split_h*lambdav*dd1x_r(rr));
              }
            }
          }
        }


        blas.matmul_transb(Nhat,Txc,Lhat);

        // Half step -- Exact solution

        blas.matmul(Lhat,Tyc,Nhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
                complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split_h*lambdaw*dd1y_r(rr))*ncvv;
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split_h*lambdaw*dd1y_r(rr))*ncvv;
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Nhat,Tyc,Lhat);

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

        // Full step -- exponential integrator
        fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

        blas.matmul(Lhat,Tzc,Nhat);

        for(Index jj = 0; jj < nsteps_ee; jj++){

          ptw_mult_row(lr_sol.V,v,Lv);
          ptw_mult_row(lr_sol.V,w,Lw);
          ptw_mult_row(lr_sol.V,u,Lu);

          fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lu.begin(),(fftw_complex*)Luhat.begin());

          blas.matmul_transb(Lvhat,D2xc,Lhat);

          blas.matmul_transb(Lwhat,D2yc,tmpVhat);

          Lhat += tmpVhat;

          blas.matmul_transb(Luhat,D2zc,tmpVhat);

          Lhat += tmpVhat;

          blas.matmul(Lhat,Tzc,tmpVhat);

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }

                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) *= exp(tau_ee*lambdau*dd1z_r(rr));
                  Nhat(idx,rr) -= tau_ee*phi1_im(tau_ee*lambdau*dd1z_r(rr))*tmpVhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) *= exp(tau_ee*lambdau*dd1z_r(rr));
                  Nhat(idx,rr) -= tau_ee*phi1_im(tau_ee*lambdau*dd1z_r(rr))*tmpVhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Nhat,Tzc,Lhat);

          Lhat *= ncvv;

          fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

          // Second stage

          ptw_mult_row(lr_sol.V,v,Lv);
          ptw_mult_row(lr_sol.V,w,Lw);
          ptw_mult_row(lr_sol.V,u,Lu);

          fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());
          fftw_execute_dft_r2c(plans_vv[0],Lu.begin(),(fftw_complex*)Luhat.begin());

          blas.matmul_transb(Lvhat,D2xc,Lhat);
          blas.matmul_transb(Lwhat,D2yc,Lvhat);

          Lvhat += Lhat;

          blas.matmul_transb(Luhat,D2zc,Lhat);

          Lvhat += Lhat;

          blas.matmul(Lvhat,Tzc,Lhat);

          Lhat -= tmpVhat;

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }

                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) -= tau_ee*phi2_im(tau_ee*lambdau*dd1z_r(rr))*Lhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_vv[2]; k++){
              if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
              for(Index j = 0; j < N_vv[1]; j++){
                for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                  complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                  Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                  Nhat(idx,rr) -= tau_ee*phi2_im(tau_ee*lambdau*dd1z_r(rr))*Lhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Nhat,Tzc,Lhat);

          if(jj != (nsteps_ee - 1)){
            Lhat *= ncvv;
            fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());
          }
        }

        // Half step -- exact solution

        blas.matmul(Lhat,Tyc,Nhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
                complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split_h*lambdaw*dd1y_r(rr));
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              if(j < (N_vv[1]/2)) { mult_j = j; } else if(j == (N_vv[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_vv[1]); }
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult_j);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split_h*lambdaw*dd1y_r(rr));
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Nhat,Tyc,Lhat);

        blas.matmul(Lhat,Txc,Nhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(tau_split_h*lambdav*dd1x_r(rr))*ncvv;
              }
            }
          }
        }

        blas.matmul_transb(Nhat,Txc,Lhat);

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      }

      gs(lr_sol.V, lr_sol.S, ip_vv);
      transpose_inplace(lr_sol.S);

      gt::stop("Full step L step CPU");

      // Half step S (until tau/2)

      gt::start("C coeff CPU");
      coeff(lr_sol.V, lr_sol.V, we_v, C1v, blas);
      coeff(lr_sol.V, lr_sol.V, we_w, C1w, blas);
      coeff(lr_sol.V, lr_sol.V, we_u, C1u, blas);

      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)tmpVhat.begin());

      ptw_mult_row(tmpVhat,lambdav_n,dVhat_v);
      ptw_mult_row(tmpVhat,lambdaw_n,dVhat_w);
      ptw_mult_row(tmpVhat,lambdau_n,dVhat_u);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_v.begin(),dV_v.begin());
      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_w.begin(),dV_w.begin());
      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_u.begin(),dV_u.begin());

      coeff(lr_sol.V, dV_v, h_vv[0]*h_vv[1]*h_vv[2], C2v, blas);
      coeff(lr_sol.V, dV_w, h_vv[0]*h_vv[1]*h_vv[2], C2w, blas);
      coeff(lr_sol.V, dV_u, h_vv[0]*h_vv[1]*h_vv[2], C2u, blas);

      gt::stop("C coeff CPU");

      gt::start("Second half step S step CPU");

      // Rk4
      for(Index jj = 0; jj< nsteps_rk4; jj++){
        blas.matmul_transb(lr_sol.S,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS1);
        blas.matmul_transb(lr_sol.S,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS1 += Tw;
        blas.matmul_transb(lr_sol.S,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS1 += Tw;
        blas.matmul_transb(lr_sol.S,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS1 -= Tw;
        blas.matmul_transb(lr_sol.S,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS1 -= Tw;
        blas.matmul_transb(lr_sol.S,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS1 -= Tw;

        Tv = tmpS1;
        Tv *= (tau_rk4_h/2);
        Tv += lr_sol.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS2);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS2 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS2 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS2 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS2 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS2 -= Tw;

        Tv = tmpS2;
        Tv *= (tau_rk4_h/2);
        Tv += lr_sol.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS3);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS3 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS3 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS3 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS3 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS3 -= Tw;

        Tv = tmpS3;
        Tv *= tau_rk4_h;
        Tv += lr_sol.S;

        blas.matmul_transb(Tv,C1v,tmpS);
        blas.matmul(D2x,tmpS,tmpS4);
        blas.matmul_transb(Tv,C1w,tmpS);
        blas.matmul(D2y,tmpS,Tw);
        tmpS4 += Tw;
        blas.matmul_transb(Tv,C1u,tmpS);
        blas.matmul(D2z,tmpS,Tw);
        tmpS4 += Tw;
        blas.matmul_transb(Tv,C2v,tmpS);
        blas.matmul(D1x,tmpS,Tw);
        tmpS4 -= Tw;
        blas.matmul_transb(Tv,C2w,tmpS);
        blas.matmul(D1y,tmpS,Tw);
        tmpS4 -= Tw;
        blas.matmul_transb(Tv,C2u,tmpS);
        blas.matmul(D1z,tmpS,Tw);
        tmpS4 -= Tw;

        tmpS2 *= 2.0;
        tmpS3 *= 2.0;

        tmpS1 += tmpS2;
        tmpS1 += tmpS3;
        tmpS1 += tmpS4;
        tmpS1 *= (tau_rk4_h/6.0);

        lr_sol.S += tmpS1;

      }

      gt::stop("Second half step S step CPU");

      // Half step K (until tau/2)

      tmpX = lr_sol.X;
      blas.matmul(tmpX,lr_sol.S,lr_sol.X);

      //gt::start("Schur K CPU");
      schur(C1v, Tv, dcv_r);
      schur(C1w, Tw, dcw_r);
      schur(C1u, Tu, dcu_r);

      Tv.to_cplx(Tvc);
      Tw.to_cplx(Twc);
      Tu.to_cplx(Tuc);
      C2v.to_cplx(C2vc);
      C2w.to_cplx(C2wc);
      C2u.to_cplx(C2uc);

      gt::start("Second half step K step CPU");

      // Internal splitting
      for(Index ii = 0; ii < nsteps_split; ii++){
        // Half step -- Exact solution

        fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

        blas.matmul(Khat,Tvc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambdax*dcv_r(rr));
              }
            }
          }
        }

        blas.matmul_transb(Mhat,Tvc,Khat);

        // Half step -- Exact solution

        blas.matmul(Khat,Twc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambday*dcw_r(rr))*ncxx;
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambday*dcw_r(rr))*ncxx;
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Mhat,Twc,Khat);

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

        // Full step -- exponential integrator

        fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

        blas.matmul(Khat,Tuc,Mhat);

        for(Index jj = 0; jj < nsteps_ee; jj++){

          ptw_mult_row(lr_sol.X,efx,Kex);
          ptw_mult_row(lr_sol.X,efy,Key);
          ptw_mult_row(lr_sol.X,efz,Kez);

          fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Kez.begin(),(fftw_complex*)Kezhat.begin());

          blas.matmul_transb(Kexhat,C2vc,Khat);
          blas.matmul_transb(Keyhat,C2wc,tmpXhat);

          Khat += tmpXhat;

          blas.matmul_transb(Kezhat,C2uc,tmpXhat);

          Khat += tmpXhat;

          //gt::start("EE Third split K CPU");
          blas.matmul(Khat,Tuc,tmpXhat);

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }

                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) *= exp(-tau_ee_h*lambdaz*dcu_r(rr));
                  Mhat(idx,rr) += tau_ee_h*phi1_im(-tau_ee_h*lambdaz*dcu_r(rr))*tmpXhat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) *= exp(-tau_ee_h*lambdaz*dcu_r(rr));
                  Mhat(idx,rr) += tau_ee_h*phi1_im(-tau_ee_h*lambdaz*dcu_r(rr))*tmpXhat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Mhat,Tuc,Khat);

          Khat *= ncxx;

          fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

          // Second stage

          ptw_mult_row(lr_sol.X,efx,Kex);
          ptw_mult_row(lr_sol.X,efy,Key);
          ptw_mult_row(lr_sol.X,efz,Kez);

          fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());
          fftw_execute_dft_r2c(plans_xx[0],Kez.begin(),(fftw_complex*)Kezhat.begin());

          blas.matmul_transb(Kexhat,C2vc,Khat);
          blas.matmul_transb(Keyhat,C2wc,Kexhat);

          Kexhat += Khat;

          blas.matmul_transb(Kezhat,C2uc,Khat);

          Kexhat += Khat;

          blas.matmul(Kexhat,Tuc,Khat);

          Khat -= tmpXhat;

          #ifdef __OPENMP__
          #pragma omp parallel for collapse(2)
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }

                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) += tau_ee_h*phi2_im(-tau_ee_h*lambdaz*dcu_r(rr))*Khat(idx,rr);
                }
              }
            }
          }
          #else
          for(int rr = 0; rr < r; rr++){
            for(Index k = 0; k < N_xx[2]; k++){
              if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
              for(Index j = 0; j < N_xx[1]; j++){
                for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                  complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                  Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                  Mhat(idx,rr) += tau_ee_h*phi2_im(-tau_ee_h*lambdaz*dcu_r(rr))*Khat(idx,rr);
                }
              }
            }
          }
          #endif

          blas.matmul_transb(Mhat,Tuc,Khat);

          if(jj != (nsteps_ee -1)){
            Khat *= ncxx;
            fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());
          }
        }

        // Half step -- exact solution

        blas.matmul(Khat,Twc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambday*dcw_r(rr));
              }
            }
          }
        }
        #else
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              if(j < (N_xx[1]/2)) { mult_j = j; } else if(j == (N_xx[1]/2)) { mult_j = 0.0; } else { mult_j = (j-N_xx[1]); }
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult_j);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambday*dcw_r(rr));
              }
            }
          }
        }
        #endif

        blas.matmul_transb(Mhat,Twc,Khat);

        // Half step -- Exact solution

        blas.matmul(Khat,Tvc,Mhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        #endif
        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);
                Mhat(idx,rr) *= exp(-tau_split_h/2.0*lambdax*dcv_r(rr))*ncxx;
              }
            }
          }
        }

        blas.matmul_transb(Mhat,Tvc,Khat);

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

      }

      gs(lr_sol.X, lr_sol.S, ip_xx);

      gt::stop("Second half step K step CPU");

      gt::stop("Restarted integration CPU");

      gt::stop("Total time CPU");
      // Error Mass
      integrate(lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],int_x,blas);
      integrate(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],int_v,blas);

      blas.matvec(lr_sol.S,int_v,rho);

      mass = 0.0;
      for(int ii = 0; ii < r; ii++){
        mass += (int_x(ii)*rho(ii));
      }

      err_mass = abs(mass0-mass);

      err_massf << err_mass << endl;

      // Error in energy

      integrate(lr_sol.V,we_v2,int_v,blas);
      integrate(lr_sol.V,we_w2,int_v2,blas);

      int_v += int_v2;

      integrate(lr_sol.V,we_u2,int_v2,blas);

      int_v += int_v2;

      blas.matvec(lr_sol.S,int_v,rho);

      energy = el_energy;
      for(int ii = 0; ii < r; ii++){
        energy += 0.5*(int_x(ii)*rho(ii));
      }

      err_energy = abs(energy0-energy);

      err_energyf << err_energy << endl;
    }

    // GPU

    #ifdef __CUDACC__

    gt::start("Total time GPU");
    /* Lie splitting in order to obtain the electric field */

    d_lr_sol_e.X = d_lr_sol.X;
    d_lr_sol_e.S = d_lr_sol.S;
    d_lr_sol_e.V = d_lr_sol.V;

    // Full step K until tau/2

    d_tmpX = d_lr_sol_e.X;

    blas.matmul(d_tmpX,d_lr_sol_e.S,d_lr_sol_e.X);

    gt::start("Lie splitting for electric field GPU");

    // Electric field

    integrate(d_lr_sol_e.V,-h_vv[0]*h_vv[1]*h_vv[2],d_rho,blas);
    blas.matvec(d_lr_sol_e.X,d_rho,d_ef);
    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier_3d<<<(dxxh_mult+n_threads-1)/n_threads,n_threads>>>(dxxh_mult, N_xx[0]/2+1, N_xx[1], N_xx[2], d_efhat.begin(), d_lim_xx, ncxx, d_efhatx.begin(), d_efhaty.begin(), d_efhatz.begin());

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatx.begin(),d_efx.begin());
    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhaty.begin(),d_efy.begin());
    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatz.begin(),d_efz.begin());

    // Electric energy

    cublasDdot (blas.handle_devres, d_efx.num_elements(), d_efx.begin(), 1, d_efx.begin(), 1, d_el_energy_x);
    cublasDdot (blas.handle_devres, d_efy.num_elements(), d_efy.begin(), 1, d_efy.begin(), 1, d_el_energy_y);
    cublasDdot (blas.handle_devres, d_efz.num_elements(), d_efz.begin(), 1, d_efz.begin(), 1, d_el_energy_z);
    cudaDeviceSynchronize();
    ptw_sum<<<1,1>>>(1,d_el_energy_x,d_el_energy_y);
    ptw_sum<<<1,1>>>(1,d_el_energy_x,d_el_energy_z);

    scale_unique<<<1,1>>>(d_el_energy_x,0.5*h_xx[0]*h_xx[1]*h_xx[2]);

    cudaMemcpy(&d_el_energy_CPU,d_el_energy_x,sizeof(double),cudaMemcpyDeviceToHost);

    el_energyGPUf << d_el_energy_CPU << endl;

    // Main of K step

    coeff(d_lr_sol_e.V, d_lr_sol_e.V, d_we_v, d_C1v, blas);
    coeff(d_lr_sol_e.V, d_lr_sol_e.V, d_we_w, d_C1w, blas);
    coeff(d_lr_sol_e.V, d_lr_sol_e.V, d_we_u, d_C1u, blas);

    cufftExecD2Z(d_plans_vv[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_tmpVhat.begin());

    ptw_mult_row_cplx_fourier_3d<<<(dvvh_mult*r+n_threads-1)/n_threads,n_threads>>>(dvvh_mult*r, N_vv[0]/2+1, N_vv[1], N_vv[2], d_tmpVhat.begin(), d_lim_vv, ncvv, d_dVhat_v.begin(), d_dVhat_w.begin(), d_dVhat_u.begin());

    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_v.begin(),d_dV_v.begin());
    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_w.begin(),d_dV_w.begin());
    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_u.begin(),d_dV_u.begin());

    coeff(d_lr_sol_e.V, d_dV_v, h_vv[0]*h_vv[1]*h_vv[2], d_C2v, blas);
    coeff(d_lr_sol_e.V, d_dV_w, h_vv[0]*h_vv[1]*h_vv[2], d_C2w, blas);
    coeff(d_lr_sol_e.V, d_dV_u, h_vv[0]*h_vv[1]*h_vv[2], d_C2u, blas);

    C1v_gpu = d_C1v;
    schur(C1v_gpu, Tv_gpu, dcv_r_gpu);
    d_Tv = Tv_gpu;
    d_dcv_r = dcv_r_gpu;
    C1w_gpu = d_C1w;
    schur(C1w_gpu, Tw_gpu, dcw_r_gpu);
    d_Tw = Tw_gpu;
    d_dcw_r = dcw_r_gpu;
    C1u_gpu = d_C1u;
    schur(C1u_gpu, Tu_gpu, dcu_r_gpu);
    d_Tu = Tu_gpu;
    d_dcu_r = dcu_r_gpu;

    cplx_conv<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(), d_Tv.begin(), d_Tvc.begin());
    cplx_conv<<<(d_Tw.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tw.num_elements(), d_Tw.begin(), d_Twc.begin());
    cplx_conv<<<(d_Tu.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tu.num_elements(), d_Tu.begin(), d_Tuc.begin());

    cplx_conv<<<(d_C2v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2v.num_elements(), d_C2v.begin(), d_C2vc.begin());
    cplx_conv<<<(d_C2w.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2w.num_elements(), d_C2w.begin(), d_C2wc.begin());
    cplx_conv<<<(d_C2u.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2u.num_elements(), d_C2u.begin(), d_C2uc.begin());

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution

      cufftExecD2Z(d_plans_xx[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

      blas.matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_3d_a<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h, d_lim_xx);

      blas.matmul_transb(d_Mhat,d_Tvc,d_Khat);

      // Full step -- Exact solution

      blas.matmul(d_Khat,d_Twc,d_Mhat);

      exact_sol_exp_3d_b<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcw_r.begin(), tau_split_h, d_lim_xx, ncxx);

      blas.matmul_transb(d_Mhat,d_Twc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol_e.X.begin());

      // Full step --

      //#ifdef __FFTW__ // working on the server
      //ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);
      //#else
      cufftExecD2Z(d_plans_xx[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_Khat.begin());
      //#endif

      blas.matmul(d_Khat,d_Tuc,d_Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efy.begin(),d_Key.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efz.begin(),d_Kez.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Kez.begin(),(cufftDoubleComplex*)d_Kezhat.begin());

        blas.matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        blas.matmul_transb(d_Keyhat,d_C2wc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        blas.matmul_transb(d_Kezhat,d_C2uc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        blas.matmul(d_Khat,d_Tuc,d_tmpXhat);

        exp_euler_fourier_3d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(),d_dcu_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin());

        blas.matmul_transb(d_Mhat,d_Tuc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol_e.X.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efy.begin(),d_Key.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efz.begin(),d_Kez.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Kez.begin(),(cufftDoubleComplex*)d_Kezhat.begin());

        blas.matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        blas.matmul_transb(d_Keyhat,d_C2wc,d_Kexhat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        blas.matmul_transb(d_Kezhat,d_C2uc,d_Khat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        blas.matmul(d_Kexhat,d_Tuc,d_Khat);

        second_ord_stage_fourier_3d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(),d_dcu_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin(), d_Khat.begin());

        blas.matmul_transb(d_Mhat,d_Tuc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol_e.X.begin());

      }
    }

    gs(d_lr_sol_e.X, d_lr_sol_e.S, h_xx[0]*h_xx[1]*h_xx[2]);

    // Full step S until tau/2

    ptw_mult_scal<<<(d_efx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efx.num_elements(), d_efx.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_x.begin());
    ptw_mult_scal<<<(d_efy.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efy.num_elements(), d_efy.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_y.begin());
    ptw_mult_scal<<<(d_efz.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efz.num_elements(), d_efz.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_z.begin());

    coeff(d_lr_sol_e.X, d_lr_sol_e.X, d_we_x, d_D1x, blas);
    coeff(d_lr_sol_e.X, d_lr_sol_e.X, d_we_y, d_D1y, blas);
    coeff(d_lr_sol_e.X, d_lr_sol_e.X, d_we_z, d_D1z, blas);

    cufftExecD2Z(d_plans_xx[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_tmpXhat.begin());

    ptw_mult_row_cplx_fourier_3d<<<(dxxh_mult*r+n_threads-1)/n_threads,n_threads>>>(dxxh_mult*r, N_xx[0]/2+1, N_xx[1], N_xx[2], d_tmpXhat.begin(), d_lim_xx, ncxx, d_dXhat_x.begin(), d_dXhat_y.begin(), d_dXhat_z.begin());

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_x.begin(),d_dX_x.begin());
    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_y.begin(),d_dX_y.begin());
    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_z.begin(),d_dX_z.begin());

    coeff(d_lr_sol_e.X, d_dX_x, h_xx[0]*h_xx[1]*h_xx[2], d_D2x, blas);
    coeff(d_lr_sol_e.X, d_dX_y, h_xx[0]*h_xx[1]*h_xx[2], d_D2y, blas);
    coeff(d_lr_sol_e.X, d_dX_z, h_xx[0]*h_xx[1]*h_xx[2], d_D2z, blas);

    // RK4
    for(Index jj = 0; jj< nsteps_split; jj++){
      blas.matmul_transb(d_lr_sol_e.S,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS1);
      blas.matmul_transb(d_lr_sol_e.S,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol_e.S,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol_e.S,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol_e.S,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol_e.S,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol_e.S.begin(), d_tmpS1.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS2);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol_e.S.begin(), d_tmpS2.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS3);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(), d_Tv.begin(), tau_rk4_h, d_lr_sol_e.S.begin(), d_tmpS3.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS4);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4_finalcomb<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(), d_lr_sol_e.S.begin(), tau_rk4_h, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_Tw.begin());

    }

    // Full step L until tau/2

    d_tmpV = d_lr_sol_e.V;

    blas.matmul_transb(d_tmpV,d_lr_sol_e.S,d_lr_sol_e.V);

    D1x_gpu = d_D1x;
    schur(D1x_gpu, Tx_gpu, dd1x_r_gpu);
    d_Tx = Tx_gpu;
    d_dd1x_r = dd1x_r_gpu;

    D1y_gpu = d_D1y;
    schur(D1y_gpu, Ty_gpu, dd1y_r_gpu);
    d_Ty = Ty_gpu;
    d_dd1y_r = dd1y_r_gpu;

    D1z_gpu = d_D1z;
    schur(D1z_gpu, Tz_gpu, dd1z_r_gpu);
    d_Tz = Tz_gpu;
    d_dd1z_r = dd1z_r_gpu;

    cplx_conv<<<(d_Tx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tx.num_elements(), d_Tx.begin(), d_Txc.begin());
    cplx_conv<<<(d_Ty.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Ty.num_elements(), d_Ty.begin(), d_Tyc.begin());
    cplx_conv<<<(d_Tz.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tz.num_elements(), d_Tz.begin(), d_Tzc.begin());

    cplx_conv<<<(d_D2x.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2x.num_elements(), d_D2x.begin(), d_D2xc.begin());
    cplx_conv<<<(d_D2y.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2y.num_elements(), d_D2y.begin(), d_D2yc.begin());
    cplx_conv<<<(d_D2z.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2z.num_elements(), d_D2z.begin(), d_D2zc.begin());

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){

      // Full step -- Exact solution
      cufftExecD2Z(d_plans_vv[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

      blas.matmul(d_Lhat,d_Txc,d_Nhat);

      exact_sol_exp_3d_a<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dd1x_r.begin(), -tau_split_h, d_lim_vv);

      blas.matmul_transb(d_Nhat,d_Txc,d_Lhat);

      blas.matmul(d_Lhat,d_Tyc,d_Nhat);

      exact_sol_exp_3d_b<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dd1y_r.begin(), -tau_split_h, d_lim_vv, ncvv);

      blas.matmul_transb(d_Nhat,d_Tyc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol_e.V.begin());

      // Full step --

      //#ifdef __FFTW__ // working on the server
      //ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), 1.0/ncvv);
      //#else
      cufftExecD2Z(d_plans_vv[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());
      //#endif

      blas.matmul(d_Lhat,d_Tzc,d_Nhat);


      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_w.begin(),d_Lw.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_u.begin(),d_Lu.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lu.begin(),(cufftDoubleComplex*)d_Luhat.begin());

        blas.matmul_transb(d_Lvhat,d_D2xc,d_Lhat);

        blas.matmul_transb(d_Lwhat,d_D2yc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        blas.matmul_transb(d_Luhat,d_D2zc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        blas.matmul(d_Lhat,d_Tzc,d_tmpVhat);

        exp_euler_fourier_3d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(),d_dd1z_r.begin(),-tau_ee_h, d_lim_vv, d_tmpVhat.begin());

        blas.matmul_transb(d_Nhat,d_Tzc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol_e.V.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_w.begin(),d_Lw.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_u.begin(),d_Lu.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lu.begin(),(cufftDoubleComplex*)d_Luhat.begin());

        blas.matmul_transb(d_Lvhat,d_D2xc,d_Lhat);
        blas.matmul_transb(d_Lwhat,d_D2yc,d_Lvhat);

        ptw_sum_complex<<<(d_Lvhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lvhat.num_elements(), d_Lvhat.begin(), d_Lhat.begin());

        blas.matmul_transb(d_Luhat,d_D2zc,d_Lhat);

        ptw_sum_complex<<<(d_Lvhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lvhat.num_elements(), d_Lvhat.begin(), d_Lhat.begin());

        blas.matmul(d_Lvhat,d_Tzc,d_Lhat);

        second_ord_stage_fourier_3d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(),d_dd1z_r.begin(),-tau_ee_h, d_lim_vv, d_tmpVhat.begin(), d_Lhat.begin());

        blas.matmul_transb(d_Nhat,d_Tzc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol_e.V.begin());


      }
    }

    cudaDeviceSynchronize();
    gt::stop("Lie splitting for electric field GPU");

    gt::start("Restarted integration GPU");

    // Electric field at time tau/2

    gt::start("Electric Field GPU");

    integrate(d_lr_sol_e.V,-h_vv[0]*h_vv[1]*h_vv[2],d_rho,blas);
    blas.matvec(d_lr_sol_e.X,d_rho,d_ef);
    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier_3d<<<(dxxh_mult+n_threads-1)/n_threads,n_threads>>>(dxxh_mult, N_xx[0]/2+1, N_xx[1], N_xx[2], d_efhat.begin(), d_lim_xx, ncxx, d_efhatx.begin(), d_efhaty.begin(), d_efhatz.begin());

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatx.begin(),d_efx.begin());
    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhaty.begin(),d_efy.begin());
    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatz.begin(),d_efz.begin());

    cudaDeviceSynchronize();
    gt::stop("Electric Field GPU");

    // Here I have the electric field at time tau/2, so restart integration

    // Half step K (until tau/2)

    d_tmpX = d_lr_sol.X;
    blas.matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    gt::start("First half step K step GPU");

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Half step -- Exact solution

      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

      blas.matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_3d_a<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h/2.0, d_lim_xx);

      blas.matmul_transb(d_Mhat,d_Tvc,d_Khat);

      // Half step -- Exact solution

      blas.matmul(d_Khat,d_Twc,d_Mhat);

      exact_sol_exp_3d_b<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcw_r.begin(), tau_split_h/2.0, d_lim_xx, ncxx);

      blas.matmul_transb(d_Mhat,d_Twc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      // Full step -- exponential integrator

      //#ifdef __FFTW__ // working on the server
      //ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);
      //#else
      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());
      //#endif

      blas.matmul(d_Khat,d_Tuc,d_Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efz.begin(),d_Kez.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Kez.begin(),(cufftDoubleComplex*)d_Kezhat.begin());

        blas.matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        blas.matmul_transb(d_Keyhat,d_C2wc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        blas.matmul_transb(d_Kezhat,d_C2uc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        blas.matmul(d_Khat,d_Tuc,d_tmpXhat);

        exp_euler_fourier_3d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(),d_dcu_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin());

        blas.matmul_transb(d_Mhat,d_Tuc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efz.begin(),d_Kez.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Kez.begin(),(cufftDoubleComplex*)d_Kezhat.begin());

        blas.matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        blas.matmul_transb(d_Keyhat,d_C2wc,d_Kexhat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        blas.matmul_transb(d_Kezhat,d_C2uc,d_Khat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        blas.matmul(d_Kexhat,d_Tuc,d_Khat);

        second_ord_stage_fourier_3d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(),d_dcu_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin(), d_Khat.begin());

        blas.matmul_transb(d_Mhat,d_Tuc,d_Khat);

        if(jj != (nsteps_ee -1)){
          ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);
          cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());
        }
      }

      // Half step -- Exact solution

      blas.matmul(d_Khat,d_Twc,d_Mhat);

      exact_sol_exp_3d_c<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcw_r.begin(), tau_split_h/2.0, d_lim_xx);

      blas.matmul_transb(d_Mhat,d_Twc,d_Khat);

      // Half step -- Exact solution

      blas.matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_3d_d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h/2.0, d_lim_xx, ncxx);

      blas.matmul_transb(d_Mhat,d_Tvc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin()); // in gpu remember to divide so we avoid the last fft

    }

    gs(d_lr_sol.X, d_lr_sol.S, h_xx[0]*h_xx[1]*h_xx[2]);

    cudaDeviceSynchronize();
    gt::stop("First half step K step GPU");

    // Half step S (until tau/2)

    gt::start("D coeff GPU");
    ptw_mult_scal<<<(d_efx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efx.num_elements(), d_efx.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_x.begin());
    ptw_mult_scal<<<(d_efy.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efy.num_elements(), d_efy.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_y.begin());
    ptw_mult_scal<<<(d_efz.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efz.num_elements(), d_efz.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_z.begin());

    coeff(d_lr_sol.X, d_lr_sol.X, d_we_x, d_D1x, blas);
    coeff(d_lr_sol.X, d_lr_sol.X, d_we_y, d_D1y, blas);
    coeff(d_lr_sol.X, d_lr_sol.X, d_we_z, d_D1z, blas);

    cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_tmpXhat.begin());

    ptw_mult_row_cplx_fourier_3d<<<(dxxh_mult*r+n_threads-1)/n_threads,n_threads>>>(dxxh_mult*r, N_xx[0]/2+1, N_xx[1], N_xx[2], d_tmpXhat.begin(), d_lim_xx, ncxx, d_dXhat_x.begin(), d_dXhat_y.begin(), d_dXhat_z.begin());

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_x.begin(),d_dX_x.begin());
    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_y.begin(),d_dX_y.begin());
    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_z.begin(),d_dX_z.begin());

    coeff(d_lr_sol.X, d_dX_x, h_xx[0]*h_xx[1]*h_xx[2], d_D2x, blas);
    coeff(d_lr_sol.X, d_dX_y, h_xx[0]*h_xx[1]*h_xx[2], d_D2y, blas);
    coeff(d_lr_sol.X, d_dX_z, h_xx[0]*h_xx[1]*h_xx[2], d_D2z, blas);

    cudaDeviceSynchronize();
    gt::stop("D coeff GPU");

    gt::start("First half step S step GPU");
    // RK4
    for(Index jj = 0; jj< nsteps_split; jj++){
      blas.matmul_transb(d_lr_sol.S,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS1);
      blas.matmul_transb(d_lr_sol.S,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS1.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS2);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS2.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS3);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(), d_Tv.begin(), tau_rk4_h, d_lr_sol.S.begin(), d_tmpS3.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS4);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4_finalcomb<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(), d_lr_sol.S.begin(), tau_rk4_h, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_Tw.begin());

    }

    cudaDeviceSynchronize();
    gt::stop("First half step S step GPU");

    // Full step L (until tau)

    d_tmpV = d_lr_sol.V;

    blas.matmul_transb(d_tmpV,d_lr_sol.S,d_lr_sol.V);

    D1x_gpu = d_D1x;
    schur(D1x_gpu, Tx_gpu, dd1x_r_gpu);
    d_Tx = Tx_gpu;
    d_dd1x_r = dd1x_r_gpu;

    D1y_gpu = d_D1y;
    schur(D1y_gpu, Ty_gpu, dd1y_r_gpu);
    d_Ty = Ty_gpu;
    d_dd1y_r = dd1y_r_gpu;

    D1z_gpu = d_D1z;
    schur(D1z_gpu, Tz_gpu, dd1z_r_gpu);
    d_Tz = Tz_gpu;
    d_dd1z_r = dd1z_r_gpu;

    cplx_conv<<<(d_Tx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tx.num_elements(), d_Tx.begin(), d_Txc.begin());
    cplx_conv<<<(d_Ty.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Ty.num_elements(), d_Ty.begin(), d_Tyc.begin());
    cplx_conv<<<(d_Tz.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tz.num_elements(), d_Tz.begin(), d_Tzc.begin());

    cplx_conv<<<(d_D2x.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2x.num_elements(), d_D2x.begin(), d_D2xc.begin());
    cplx_conv<<<(d_D2y.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2y.num_elements(), d_D2y.begin(), d_D2yc.begin());
    cplx_conv<<<(d_D2z.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2z.num_elements(), d_D2z.begin(), d_D2zc.begin());

    gt::start("Full step L step GPU");
    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){

      // Half step -- Exact solution
      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

      blas.matmul(d_Lhat,d_Txc,d_Nhat);

      exact_sol_exp_3d_a<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dd1x_r.begin(), -tau_split_h, d_lim_vv);

      blas.matmul_transb(d_Nhat,d_Txc,d_Lhat);

      // Half step -- Exact solution

      blas.matmul(d_Lhat,d_Tyc,d_Nhat);

      exact_sol_exp_3d_b<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dd1y_r.begin(), -tau_split_h, d_lim_vv, ncvv);

      blas.matmul_transb(d_Nhat,d_Tyc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      // Full step -- exponential integrator

      //#ifdef __FFTW__ // working on the server
      //ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), 1.0/ncvv);
      //#else
      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());
      //#endif

      blas.matmul(d_Lhat,d_Tzc,d_Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_u.begin(),d_Lu.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lu.begin(),(cufftDoubleComplex*)d_Luhat.begin());

        blas.matmul_transb(d_Lvhat,d_D2xc,d_Lhat);

        blas.matmul_transb(d_Lwhat,d_D2yc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        blas.matmul_transb(d_Luhat,d_D2zc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        blas.matmul(d_Lhat,d_Tzc,d_tmpVhat);

        exp_euler_fourier_3d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(),d_dd1z_r.begin(),-tau_ee, d_lim_vv, d_tmpVhat.begin());

        blas.matmul_transb(d_Nhat,d_Tzc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_u.begin(),d_Lu.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin());
        cufftExecD2Z(d_plans_vv[0],d_Lu.begin(),(cufftDoubleComplex*)d_Luhat.begin());

        blas.matmul_transb(d_Lvhat,d_D2xc,d_Lhat);
        blas.matmul_transb(d_Lwhat,d_D2yc,d_Lvhat);

        ptw_sum_complex<<<(d_Lvhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lvhat.num_elements(), d_Lvhat.begin(), d_Lhat.begin());

        blas.matmul_transb(d_Luhat,d_D2zc,d_Lhat);

        ptw_sum_complex<<<(d_Lvhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lvhat.num_elements(), d_Lvhat.begin(), d_Lhat.begin());

        blas.matmul(d_Lvhat,d_Tzc,d_Lhat);

        second_ord_stage_fourier_3d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(),d_dd1z_r.begin(),-tau_ee, d_lim_vv, d_tmpVhat.begin(), d_Lhat.begin());

        blas.matmul_transb(d_Nhat,d_Tzc,d_Lhat);

        if(jj != (nsteps_ee - 1)){
          ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);
          cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());
        }
      }

      // Half step -- Exact solution

      blas.matmul(d_Lhat,d_Tyc,d_Nhat);

      exact_sol_exp_3d_c<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dd1y_r.begin(), -tau_split_h, d_lim_vv);

      blas.matmul_transb(d_Nhat,d_Tyc,d_Lhat);

      // Half step -- Exact solution

      blas.matmul(d_Lhat,d_Txc,d_Nhat);

      exact_sol_exp_3d_d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dd1x_r.begin(), -tau_split_h, d_lim_vv, ncvv);

      blas.matmul_transb(d_Nhat,d_Txc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

    }

    gs(d_lr_sol.V, d_lr_sol.S, h_vv[0]*h_vv[1]*h_vv[2]);
    //transpose_inplace<<<d_lr_sol.S.num_elements(),1>>>(r,d_lr_sol.S.begin());
    transpose_inplace(d_lr_sol.S);
    cudaDeviceSynchronize();
    gt::stop("Full step L step GPU");

    // Half step S (until tau/2)

    gt::start("C coeff GPU");
    coeff(d_lr_sol.V, d_lr_sol.V, d_we_v, d_C1v, blas);
    coeff(d_lr_sol.V, d_lr_sol.V, d_we_w, d_C1w, blas);
    coeff(d_lr_sol.V, d_lr_sol.V, d_we_u, d_C1u, blas);

    cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_tmpVhat.begin());

    ptw_mult_row_cplx_fourier_3d<<<(dvvh_mult*r+n_threads-1)/n_threads,n_threads>>>(dvvh_mult*r, N_vv[0]/2+1, N_vv[1], N_vv[2], d_tmpVhat.begin(), d_lim_vv, ncvv, d_dVhat_v.begin(), d_dVhat_w.begin(), d_dVhat_u.begin());

    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_v.begin(),d_dV_v.begin());
    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_w.begin(),d_dV_w.begin());
      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_u.begin(),d_dV_u.begin());

    coeff(d_lr_sol.V, d_dV_v, h_vv[0]*h_vv[1]*h_vv[2], d_C2v, blas);
    coeff(d_lr_sol.V, d_dV_w, h_vv[0]*h_vv[1]*h_vv[2], d_C2w, blas);
    coeff(d_lr_sol.V, d_dV_u, h_vv[0]*h_vv[1]*h_vv[2], d_C2u, blas);

    cudaDeviceSynchronize();
    gt::stop("C coeff GPU");

    gt::start("Second half step S step GPU");
    // RK4
    for(Index jj = 0; jj< nsteps_split; jj++){
      blas.matmul_transb(d_lr_sol.S,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS1);
      blas.matmul_transb(d_lr_sol.S,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      blas.matmul_transb(d_lr_sol.S,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS1.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS2);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS2.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS3);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(), d_Tv.begin(), tau_rk4_h, d_lr_sol.S.begin(), d_tmpS3.begin(),d_Tw.begin());

      blas.matmul_transb(d_Tv,d_C1v,d_tmpS);
      blas.matmul(d_D2x,d_tmpS,d_tmpS4);
      blas.matmul_transb(d_Tv,d_C1w,d_tmpS);
      blas.matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C1u,d_tmpS);
      blas.matmul(d_D2z,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2v,d_tmpS);
      blas.matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2w,d_tmpS);
      blas.matmul(d_D1y,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      blas.matmul_transb(d_Tv,d_C2u,d_tmpS);
      blas.matmul(d_D1z,d_tmpS,d_Tw);

      rk4_finalcomb<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(), d_lr_sol.S.begin(), tau_rk4_h, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_Tw.begin());

    }

    cudaDeviceSynchronize();
    gt::stop("Second half step S step GPU");

    // Half step K (until tau/2)

    d_tmpX = d_lr_sol.X;
    blas.matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    C1v_gpu = d_C1v;
    schur(C1v_gpu, Tv_gpu, dcv_r_gpu);
    d_Tv = Tv_gpu;
    d_dcv_r = dcv_r_gpu;
    C1w_gpu = d_C1w;
    schur(C1w_gpu, Tw_gpu, dcw_r_gpu);
    d_Tw = Tw_gpu;
    d_dcw_r = dcw_r_gpu;
    C1u_gpu = d_C1u;
    schur(C1u_gpu, Tu_gpu, dcu_r_gpu);
    d_Tu = Tu_gpu;
    d_dcu_r = dcu_r_gpu;

    cplx_conv<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(), d_Tv.begin(), d_Tvc.begin());
    cplx_conv<<<(d_Tw.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tw.num_elements(), d_Tw.begin(), d_Twc.begin());
    cplx_conv<<<(d_Tu.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tu.num_elements(), d_Tu.begin(), d_Tuc.begin());

    cplx_conv<<<(d_C2v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2v.num_elements(), d_C2v.begin(), d_C2vc.begin());
    cplx_conv<<<(d_C2w.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2w.num_elements(), d_C2w.begin(), d_C2wc.begin());
    cplx_conv<<<(d_C2u.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2u.num_elements(), d_C2u.begin(), d_C2uc.begin());

    gt::start("Second half step K step GPU");
    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Half step -- Exact solution

      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

      blas.matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_3d_a<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h/2.0, d_lim_xx);

      blas.matmul_transb(d_Mhat,d_Tvc,d_Khat);

      // Half step -- Exact solution

      blas.matmul(d_Khat,d_Twc,d_Mhat);

      exact_sol_exp_3d_b<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcw_r.begin(), tau_split_h/2.0, d_lim_xx, ncxx);

      blas.matmul_transb(d_Mhat,d_Twc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      // Full step -- exponential integrator

      //#ifdef __FFTW__ // working on the server
      //ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);
      //#else
      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());
      //#endif

      blas.matmul(d_Khat,d_Tuc,d_Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efz.begin(),d_Kez.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Kez.begin(),(cufftDoubleComplex*)d_Kezhat.begin());

        blas.matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        blas.matmul_transb(d_Keyhat,d_C2wc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        blas.matmul_transb(d_Kezhat,d_C2uc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        blas.matmul(d_Khat,d_Tuc,d_tmpXhat);

        exp_euler_fourier_3d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(),d_dcu_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin());

        blas.matmul_transb(d_Mhat,d_Tuc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efz.begin(),d_Kez.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Kez.begin(),(cufftDoubleComplex*)d_Kezhat.begin());

        blas.matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        blas.matmul_transb(d_Keyhat,d_C2wc,d_Kexhat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        blas.matmul_transb(d_Kezhat,d_C2uc,d_Khat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        blas.matmul(d_Kexhat,d_Tuc,d_Khat);

        second_ord_stage_fourier_3d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(),d_dcu_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin(), d_Khat.begin());

        blas.matmul_transb(d_Mhat,d_Tuc,d_Khat);

        if(jj != (nsteps_ee -1)){
          ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);
          cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());
        }
      }

      // Half step -- Exact solution

      blas.matmul(d_Khat,d_Twc,d_Mhat);

      exact_sol_exp_3d_c<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcw_r.begin(), tau_split_h/2.0, d_lim_xx);

      blas.matmul_transb(d_Mhat,d_Twc,d_Khat);

      // Half step -- Exact solution

      blas.matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_3d_d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h/2.0, d_lim_xx, ncxx);

      blas.matmul_transb(d_Mhat,d_Tvc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin()); // in gpu remember to divide so we avoid the last fft

    }

    gs(d_lr_sol.X, d_lr_sol.S, h_xx[0]*h_xx[1]*h_xx[2]);

    cudaDeviceSynchronize();
    gt::stop("Second half step K step GPU");

    cudaDeviceSynchronize();
    gt::stop("Restarted integration GPU");

    cudaDeviceSynchronize();
    gt::stop("Total time GPU");
    // Error mass

    integrate(d_lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],d_int_x,blas);
    integrate(d_lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],d_int_v,blas);

    blas.matvec(d_lr_sol.S,d_int_v,d_rho);

    cublasDdot (blas.handle_devres, r, d_int_x.begin(), 1, d_rho.begin(), 1,d_mass);
    cudaDeviceSynchronize();

    cudaMemcpy(&d_mass_CPU,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    err_mass_CPU = abs(mass0-d_mass_CPU);

    err_massGPUf << err_mass_CPU << endl;

    // Error energy

    integrate(d_lr_sol.V,d_we_v2,d_int_v,blas);
    integrate(d_lr_sol.V,d_we_w2,d_int_v2,blas);
    integrate(d_lr_sol.V,d_we_u2,d_int_v3,blas);

    ptw_sum_3mat<<<(d_int_v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_int_v.num_elements(),d_int_v.begin(),d_int_v2.begin(),d_int_v3.begin());

    blas.matvec(d_lr_sol.S,d_int_v,d_rho);

    cublasDdot (blas.handle_devres, r, d_int_x.begin(), 1, d_rho.begin(), 1, d_energy);
    cudaDeviceSynchronize();
    scale_unique<<<1,1>>>(d_energy,0.5); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
    cudaMemcpy(&d_energy_CPU,d_energy,sizeof(double),cudaMemcpyDeviceToHost);

    err_energy_CPU = abs(energy0-(d_energy_CPU+d_el_energy_CPU));

    err_energyGPUf << err_energy_CPU << endl;

    #endif

    if(CPU) {
      cout << "Electric energy: " << el_energy << endl;
    } else {
      #ifdef __CUDACC__
      cout << "Electric energy GPU: " << d_el_energy_CPU << endl;
      #endif
    }

    if(CPU) {
    cout << "Error in mass: " << err_mass << endl;
    } else {
      #ifdef __CUDACC__
      cout << "Error in mass GPU: " << err_mass_CPU << endl;
      #endif
    }

    if(CPU) {
    cout << "Error in energy: " << err_energy << endl;
    } else {
      #ifdef __CUDACC__
      cout << "Error in energy GPU: " << err_energy_CPU << endl;
      #endif
    }

  }

  #ifdef __CUDACC__
  destroy_plans(plans_d_e);
  destroy_plans(d_plans_xx);
  destroy_plans(d_plans_vv);

  el_energyGPUf.close();
  err_massGPUf.close();
  err_energyGPUf.close();
  #endif

  #ifdef __CUDACC__
  lr_sol.X = d_lr_sol.X;
  lr_sol.S = d_lr_sol.S;
  lr_sol.V = d_lr_sol.V;
  cudaDeviceSynchronize();
  #endif

  return lr_sol;
}


int main(int argc, char** argv){

  cxxopts::Options options("linear_landau_3d", "3+3 dimensional simulation of linear Landau damping");
  options.add_options()
  ("cpu", "Run the simulation on the CPU", cxxopts::value<bool>()->default_value("false"))
  ("nx", "Number of grid points in space    (as a whitespace separated list)", cxxopts::value<string>()->default_value("32 32 32"))
  ("nv", "Number of grid points in velocity (as a whitespace separated list)", cxxopts::value<string>()->default_value("32 32 32"))
  ("h,help", "Help message")
  ;
  auto result = options.parse(argc, argv);

  if(result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  #ifndef __CUDACC__
  CPU = true;
  #else
  CPU = result["cpu"].as<bool>();
  #endif

  array<Index,3> N_xx = parse<3>(result["nx"].as<string>());
  array<Index,3> N_vv = parse<3>(result["nv"].as<string>());


  #ifdef __OPENMP__
  omp_set_num_threads(n_threads_omp);

  #pragma omp parallel
  {
    if(omp_get_thread_num()==0){
      cout << "Number of threads: " << omp_get_num_threads() << endl;
    }
  }
  #endif

  int r = 20; // rank desired 10

  double tstar = 60; // final time //0.05 for two stream

  Index nsteps_ref = 6000; //7000

  vector<Index> nspan = {40,50,60,70,80}; //25:25:300

  int nsteps_split = 1;
  int nsteps_ee = 1;
  int nsteps_rk4 = 1;

  // Linear Landau
  array<double,6> lim_xx = {0.0,4.0*M_PI,0.0,4.0*M_PI,0.0,4.0*M_PI}; // Limits for box [ax,bx] x [ay,by] x [az,bz] {ax,bx,ay,by,az,bz}
  array<double,6> lim_vv = {-6.0,6.0,-6.0,6.0,-6.0,6.0}; // Limits for box [av,bv] x [aw,bw] x [au,bu] {av,bv,aw,bw,au,bu}
  double alpha1 = 0.01;
  double alpha2 = 0.01;
  double alpha3 = 0.01;
  double kappa1 = 0.5;
  double kappa2 = 0.5;
  double kappa3 = 0.5;

/*
  // Two stream instability
  array<double,6> lim_xx = {0.0,10.0*M_PI,0.0,10.0*M_PI,0.0,10.0*M_PI};
  array<double,6> lim_vv = {-9.0,9.0,-9.0,9.0,-9.0,9.0};

  double alpha1 = 0.001;
  double alpha2 = 0.001;
  double alpha3 = 0.001;
  double kappa1 = 1.0/5.0;
  double kappa2 = 1.0/5.0;
  double kappa3 = 1.0/5.0;
  double v0 = 2.5;
  double w0 = 0;
  double u0 = 0;
  double v0b = -2.5;
  double w0b = -2.25;
  double u0b = -2.0;
*/
  // Initial datum generation

  gt::start("Initial datum generation (CPU)");

  array<double,3> h_xx, h_vv;
  int jj = 0;
  for(int ii = 0; ii < 3; ii++){
    h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
    h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
    jj+=2;
  }

  vector<const double*> X, V;

  Index dxx_mult = N_xx[0]*N_xx[1]*N_xx[2];
  Index dxxh_mult = N_xx[2]*N_xx[1]*(N_xx[0]/2 + 1);

  Index dvv_mult = N_vv[0]*N_vv[1]*N_vv[2];
  Index dvvh_mult = N_vv[2]*N_vv[1]*(N_vv[0]/2 + 1);

  multi_array<double,1> xx({dxx_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index k = 0; k < N_xx[2]; k++){
    for(Index j = 0; j < N_xx[1]; j++){
      for(Index i = 0; i < N_xx[0]; i++){
        double x = lim_xx[0] + i*h_xx[0];
        double y = lim_xx[2] + j*h_xx[1];
        double z = lim_xx[4] + k*h_xx[2];

        xx(i+j*N_xx[0] + k*(N_xx[0]*N_xx[1])) = 1.0 + alpha1*cos(kappa1*x) + alpha2*cos(kappa2*y) + alpha3*cos(kappa3*z);
      }
    }
  }
  X.push_back(xx.begin());

  multi_array<double,1> vv({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index k = 0; k < N_vv[2]; k++){
    for(Index j = 0; j < N_vv[1]; j++){
      for(Index i = 0; i < N_vv[0]; i++){
        Index idx = i+j*N_vv[0] + k*(N_vv[0]*N_vv[1]);
        double v = lim_vv[0] + i*h_vv[0];
        double w = lim_vv[2] + j*h_vv[1];
        double u = lim_vv[4] + k*h_vv[2];

        vv(idx) = (1.0/(sqrt(pow(2*M_PI,3)))) * exp(-(pow(v,2)+pow(w,2)+pow(u,2))/2.0);
        //vv(idx) = (1.0/(sqrt(pow(8*M_PI,3)))) * (exp(-(pow(v-v0,2))/2.0)+exp(-(pow(v-v0b,2))/2.0))*(exp(-(pow(w-w0,2))/2.0)+exp(-(pow(w-w0b,2))/2.0))*(exp(-(pow(u-u0,2))/2.0)+exp(-(pow(u-u0b,2))/2.0));
      }
    }
  }
  V.push_back(vv.begin());

  lr2<double> lr_sol0(r,{dxx_mult,dvv_mult});

  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(h_xx[0]*h_xx[1]*h_xx[2], dxx_mult);
  std::function<double(double*,double*)> ip_vv = inner_product_from_const_weight(h_vv[0]*h_vv[1]*h_vv[2], dvv_mult);

  gt::stop("Initial datum generation (CPU)");

  // Mandatory to initialize after plan creation if we use FFTW_MEASURE

  gt::start("FFTW plans (CPU)");

  multi_array<double,1> ef({dxx_mult});
  multi_array<complex<double>,1> efhat({dxxh_mult});

  array<fftw_plan,2> plans_e = create_plans_3d(N_xx, ef, efhat);

  multi_array<complex<double>,2> Khat({dxxh_mult,r});
  array<fftw_plan,2> plans_xx = create_plans_3d(N_xx, lr_sol0.X, Khat);

  multi_array<complex<double>,2> Lhat({dvvh_mult,r});
  array<fftw_plan,2> plans_vv = create_plans_3d(N_vv, lr_sol0.V, Lhat);

  gt::stop("FFTW plans (CPU)");


  gt::start("Initial datum generation (CPU)");

  blas_ops blas;
  initialize(lr_sol0, X, V, ip_xx, ip_vv, blas); // Mandatory to initialize after plan creation as we use FFTW_MEASURE

  gt::stop("Initial datum generation (CPU)");

  // Computation of reference solution
  lr2<double> lr_sol_fin(r,{dxx_mult,dvv_mult});

  //cout << "First order" << endl;

  //lr_sol_fin = integration_first_order(N_xx,N_vv,r,tstar,nsteps_ref,nsteps_split,nsteps_ee,nsteps_rk4,lim_xx,lim_vv,lr_sol0, plans_e, plans_xx, plans_vv, blas);

  //cout << "Second order" << endl;
  lr_sol_fin = integration_second_order(N_xx,N_vv,r,tstar,nsteps_ref,nsteps_split,nsteps_ee,nsteps_rk4,lim_xx,lim_vv,lr_sol0, plans_e, plans_xx, plans_vv, blas);

  cout << gt::sorted_output() << endl;

  exit(1);
  multi_array<double,2> refsol({dxx_mult,dvv_mult});
  multi_array<double,2> sol({dxx_mult,dvv_mult});
  multi_array<double,2> tmpsol({dxx_mult,r});

  blas.matmul(lr_sol_fin.X,lr_sol_fin.S,tmpsol);

  blas.matmul_transb(tmpsol,lr_sol_fin.V,refsol);

  ofstream error_order1_3d;
  error_order1_3d.open("error_order1_3d.txt");
  error_order1_3d.precision(16);

  ofstream error_order2_3d;
  error_order2_3d.open("error_order2_3d.txt");
  error_order2_3d.precision(16);

  error_order1_3d << nspan.size() << endl;
  for(int count = 0; count < nspan.size(); count++){
    error_order1_3d << nspan[count] << endl;
  }

  error_order2_3d << nspan.size() << endl;
  for(int count = 0; count < nspan.size(); count++){
    error_order2_3d << nspan[count] << endl;
  }

  for(int count = 0; count < nspan.size(); count++){

    lr_sol_fin = integration_first_order(N_xx,N_vv,r,tstar,nspan[count],nsteps_split,nsteps_ee,nsteps_rk4,lim_xx,lim_vv,lr_sol0, plans_e, plans_xx, plans_vv, blas);

    blas.matmul(lr_sol_fin.X,lr_sol_fin.S,tmpsol);
    blas.matmul_transb(tmpsol,lr_sol_fin.V,sol);

    double error_o1 = 0.0;

    for(int iii = 0; iii < dxx_mult; iii++){
      for(int jjj = 0; jjj < dvv_mult; jjj++){
        double value = abs(refsol(iii,jjj)-sol(iii,jjj));
        if( error_o1 < value){
          error_o1 = value;
        }
      }
    }
    error_order1_3d << error_o1 << endl;

    lr_sol_fin = integration_second_order(N_xx,N_vv,r,tstar,nspan[count],nsteps_split,nsteps_ee,nsteps_rk4,lim_xx,lim_vv,lr_sol0, plans_e, plans_xx, plans_vv, blas);

    blas.matmul(lr_sol_fin.X,lr_sol_fin.S,tmpsol);
    blas.matmul_transb(tmpsol,lr_sol_fin.V,sol);

    double error_o2 = 0.0;
    for(int iii = 0; iii < dxx_mult; iii++){
      for(int jjj = 0; jjj < dvv_mult; jjj++){
        double value = abs(refsol(iii,jjj)-sol(iii,jjj));
        if( error_o2 < value){
          error_o2 = value;
        }
      }
    }
    error_order2_3d << error_o2 << endl;

  }

  error_order1_3d.close();
  error_order2_3d.close();

  destroy_plans(plans_e);
  destroy_plans(plans_xx);
  destroy_plans(plans_vv);

  return 0;
}
