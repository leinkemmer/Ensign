#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/kernels.hpp>
#include <generic/timer.hpp>

#ifdef __CUDACC__
  cublasHandle_t  handle;
  cublasHandle_t handle_dot;
#endif

lr2<double> integration_first_order(array<Index,2> N_xx,array<Index,2> N_vv, int r,double tstar, Index nsteps, int nsteps_split, int nsteps_ee, int nsteps_rk4, array<double,4> lim_xx, array<double,4> lim_vv, double alpha, double kappa1, double kappa2, lr2<double> lr_sol, array<fftw_plan,2> plans_e, array<fftw_plan,2> plans_xx, array<fftw_plan,2> plans_vv){
  double tau = tstar/nsteps;

  double tau_split = tau/nsteps_split;
  double tau_ee = tau_split / nsteps_ee;
  double tau_rk4 = tau / nsteps_rk4;

  array<double,2> h_xx, h_vv;
  int jj = 0;
  for(int ii = 0; ii < 2; ii++){
    h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
    h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
    jj+=2;
  }

  Index dxx_mult = N_xx[0]*N_xx[1];
  Index dxxh_mult = N_xx[1]*(N_xx[0]/2 + 1);

  Index dvv_mult = N_vv[0]*N_vv[1];
  Index dvvh_mult = N_vv[1]*(N_vv[0]/2 + 1);

  multi_array<double,1> v({dvv_mult});
  multi_array<double,1> w({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < N_vv[1]; j++){
    for(Index i = 0; i < N_vv[0]; i++){
      v(i+j*N_vv[0]) = lim_vv[0] + i*h_vv[0];
      w(i+j*N_vv[0]) = lim_vv[2] + j*h_vv[1];
    }
  }

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({dxx_mult});
  multi_array<double,1> efx({dxx_mult});
  multi_array<double,1> efy({dxx_mult});

  multi_array<complex<double>,1> efhat({dxxh_mult});
  multi_array<complex<double>,1> efhatx({dxxh_mult});
  multi_array<complex<double>,1> efhaty({dxxh_mult});

  // Some FFT stuff for X and V

  multi_array<complex<double>,1> lambdax_n({dxxh_mult});
  multi_array<complex<double>,1> lambday_n({dxxh_mult});

  double ncxx = 1.0 / (dxx_mult);

  Index mult;
  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index j = 0; j < N_xx[1]; j++){
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }

      lambdax_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i)*ncxx;
      lambday_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult)*ncxx;
    }
  }
  #else
  for(Index j = 0; j < N_xx[1]; j++){
    if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      lambdax_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i)*ncxx;
      lambday_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult)*ncxx;
    }
  }
  #endif

  multi_array<complex<double>,1> lambdav_n({dvvh_mult});
  multi_array<complex<double>,1> lambdaw_n({dvvh_mult});

  double ncvv = 1.0 / (dvv_mult);

  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index j = 0; j < N_vv[1]; j++){
    for(Index i = 0; i < (N_vv[0]/2+1); i++){
      if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }

      lambdav_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i)*ncvv;
      lambdaw_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult)*ncvv;
    }
  }
  #else
  for(Index j = 0; j < N_vv[1]; j++){
    if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
    for(Index i = 0; i < (N_vv[0]/2+1); i++){
      lambdav_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i)*ncvv;
      lambdaw_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult)*ncvv;
    }
  }
  #endif
  multi_array<complex<double>,2> Khat({dxxh_mult,r});

  multi_array<complex<double>,2> Lhat({dvvh_mult,r});

  // For C coefficients
  multi_array<double,2> C1v({r,r});
  multi_array<double,2> C1w({r,r});

  multi_array<double,2> C2v({r,r});
  multi_array<double,2> C2w({r,r});

  multi_array<complex<double>,2> C2vc({r,r});
  multi_array<complex<double>,2> C2wc({r,r});

  multi_array<double,1> we_v({dvv_mult});
  multi_array<double,1> we_w({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
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

  multi_array<complex<double>,2> D2xc({r,r});
  multi_array<complex<double>,2> D2yc({r,r});

  multi_array<double,1> we_x({dxx_mult});
  multi_array<double,1> we_y({dxx_mult});

  multi_array<double,2> dX_x({dxx_mult,r});
  multi_array<double,2> dX_y({dxx_mult,r});

  multi_array<complex<double>,2> dXhat_x({dxxh_mult,r});
  multi_array<complex<double>,2> dXhat_y({dxxh_mult,r});

  // For Schur decomposition
  multi_array<double,1> dcv_r({r});
  multi_array<double,1> dcw_r({r});
  multi_array<double,1> dd1x_r({r});
  multi_array<double,1> dd1y_r({r});

  multi_array<double,2> Tv({r,r});
  multi_array<double,2> Tw({r,r});
  multi_array<double,2> Tx({r,r});
  multi_array<double,2> Ty({r,r});

  multi_array<complex<double>,2> Mhat({dxxh_mult,r});
  multi_array<complex<double>,2> Nhat({dvvh_mult,r});
  multi_array<complex<double>,2> Tvc({r,r});
  multi_array<complex<double>,2> Twc({r,r});
  multi_array<complex<double>,2> Txc({r,r});
  multi_array<complex<double>,2> Tyc({r,r});

  #ifdef __MKL__
    MKL_INT lwork = -1;
  #else
    int lwork = -1;
  #endif

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
  multi_array<double,2> tmpS1({r,r});
  multi_array<double,2> tmpS2({r,r});
  multi_array<double,2> tmpS3({r,r});
  multi_array<double,2> tmpS4({r,r});

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

  // Initial mass
  integrate(lr_sol.X,h_xx[0]*h_xx[1],int_x);
  integrate(lr_sol.V,h_vv[0]*h_vv[1],int_v);

  matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    mass0 += (int_x(ii)*rho(ii));
  }

  // Initial energy
  integrate(lr_sol.V,h_vv[0]*h_vv[1],rho);
  rho *= -1.0;
  matmul(lr_sol.X,lr_sol.S,tmpX);
  matvec(tmpX,rho,ef);
  ef += 1.0;
  fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());
  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index j = 0; j < N_xx[1]; j++){
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);
      efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
      efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
    }
  }
  #else
  for(Index j = 0; j < N_xx[1]; j++){
    if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);
      efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
      efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
    }
  }
  #endif
  efhatx(0) = complex<double>(0.0,0.0);
  efhaty(0) = complex<double>(0.0,0.0);
  efhatx((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
  efhaty((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());

  energy0 = 0.0;
  #ifdef __OPENMP__
  #pragma omp parallel for reduction(+:energy0)
  #endif
  for(Index ii = 0; ii < (dxx_mult); ii++){
    energy0 += 0.5*(pow(efx(ii),2)+pow(efy(ii),2))*h_xx[0]*h_xx[1];
  }

  multi_array<double,1> we_v2({dvv_mult});
  multi_array<double,1> we_w2({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < (dvv_mult); j++){
    we_v2(j) = pow(v(j),2) * h_vv[0] * h_vv[1];
    we_w2(j) = pow(w(j),2) * h_vv[0] * h_vv[1];
  }

  integrate(lr_sol.V,we_v2,int_v);
  integrate(lr_sol.V,we_w2,int_v2);

  int_v += int_v2;

  matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    energy0 += 0.5*(int_x(ii)*rho(ii));
  }

  #ifdef __CPU__

  ofstream el_energyf;
  ofstream err_massf;
  ofstream err_energyf;

  el_energyf.open("../../plots/el_energy_order1_2d.txt");
  err_massf.open("../../plots/err_mass_order1_2d.txt");
  err_energyf.open("../../plots/err_energy_order1_2d.txt");

  el_energyf.precision(16);
  err_massf.precision(16);
  err_energyf.precision(16);

  el_energyf << tstar << endl;
  el_energyf << tau << endl;

  #endif
  //// FOR GPU ////

  #ifdef __CUDACC__
  cublasCreate (&handle);
  cublasCreate (&handle_dot);
  cublasSetPointerMode(handle_dot, CUBLAS_POINTER_MODE_DEVICE);

  // To be substituted if we initialize in GPU

  lr2<double> d_lr_sol(r,{dxx_mult,dvv_mult},stloc::device);
  d_lr_sol.X = lr_sol.X;
  d_lr_sol.V = lr_sol.V;
  d_lr_sol.S = lr_sol.S;

  // For Electric field

  multi_array<double,1> d_rho({r},stloc::device);
  multi_array<double,1> d_ef({dxx_mult},stloc::device);
  multi_array<double,1> d_efx({dxx_mult},stloc::device);
  multi_array<double,1> d_efy({dxx_mult},stloc::device);

  multi_array<cuDoubleComplex,1> d_efhat({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhatx({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhaty({dxxh_mult},stloc::device);

  array<cufftHandle,2> plans_d_e = create_plans_2d(N_xx, 1);

  // FFT stuff

  double* d_lim_xx;
  cudaMalloc((void**)&d_lim_xx,4*sizeof(double));
  cudaMemcpy(d_lim_xx,lim_xx.begin(),4*sizeof(double),cudaMemcpyHostToDevice);

  double* d_lim_vv;
  cudaMalloc((void**)&d_lim_vv,4*sizeof(double));
  cudaMemcpy(d_lim_vv,lim_vv.begin(),4*sizeof(double),cudaMemcpyHostToDevice);

  multi_array<cuDoubleComplex,2> d_Khat({dxxh_mult,r},stloc::device);
  array<cufftHandle,2> d_plans_xx = create_plans_2d(N_xx, r);

  multi_array<cuDoubleComplex,2> d_Lhat({dvvh_mult,r},stloc::device);
  array<cufftHandle,2> d_plans_vv = create_plans_2d(N_vv, r);

  // C coefficients

  multi_array<double,2> d_C1v({r,r},stloc::device);
  multi_array<double,2> d_C1w({r,r}, stloc::device);

  multi_array<double,2> d_C2v({r,r},stloc::device);
  multi_array<double,2> d_C2w({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_C2vc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2wc({r,r},stloc::device);

  multi_array<double,1> d_we_v({dvv_mult},stloc::device);
  d_we_v = we_v;

  multi_array<double,1> d_we_w({dvv_mult},stloc::device);
  d_we_w = we_w;

  multi_array<double,2> d_dV_v({dvv_mult,r},stloc::device);
  multi_array<double,2> d_dV_w({dvv_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_dVhat_v({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dVhat_w({dvvh_mult,r},stloc::device);


  // D coefficients

  multi_array<double,2> d_D1x({r,r},stloc::device);
  multi_array<double,2> d_D1y({r,r},stloc::device);

  multi_array<double,2> d_D2x({r,r},stloc::device);
  multi_array<double,2> d_D2y({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_D2xc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_D2yc({r,r},stloc::device);


  multi_array<double,1> d_we_x({dxx_mult},stloc::device);
  multi_array<double,1> d_we_y({dxx_mult},stloc::device);

  multi_array<double,2> d_dX_x({dxx_mult,r},stloc::device);
  multi_array<double,2> d_dX_y({dxx_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_dXhat_x({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dXhat_y({dxxh_mult,r},stloc::device);

  multi_array<double,1> d_v({dvv_mult},stloc::device);
  d_v = v;
  multi_array<double,1> d_w({dvv_mult},stloc::device);
  d_w = w;


  // Schur decomposition

  multi_array<double,2> C1v_gpu({r,r});
  multi_array<double,2> Tv_gpu({r,r});
  multi_array<double,1> dcv_r_gpu({r});
  multi_array<double,2> C1w_gpu({r,r});
  multi_array<double,2> Tw_gpu({r,r});
  multi_array<double,1> dcw_r_gpu({r});

  multi_array<double,2> D1x_gpu({r,r});
  multi_array<double,2> Tx_gpu({r,r});
  multi_array<double,1> dd1x_r_gpu({r});
  multi_array<double,2> D1y_gpu({r,r});
  multi_array<double,2> Ty_gpu({r,r});
  multi_array<double,1> dd1y_r_gpu({r});

  multi_array<double,1> d_dcv_r({r},stloc::device);
  multi_array<double,2> d_Tv({r,r},stloc::device);
  multi_array<double,1> d_dcw_r({r},stloc::device);
  multi_array<double,2> d_Tw({r,r},stloc::device);

  multi_array<double,1> d_dd1x_r({r},stloc::device);
  multi_array<double,2> d_Tx({r,r},stloc::device);
  multi_array<double,1> d_dd1y_r({r},stloc::device);
  multi_array<double,2> d_Ty({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Mhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Nhat({dvvh_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Tvc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Twc({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Txc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Tyc({r,r},stloc::device);

  // For K step

  multi_array<double,2> d_Kex({dxx_mult,r},stloc::device);
  multi_array<double,2> d_Key({dxx_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Kexhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Keyhat({dxxh_mult,r},stloc::device);

  // For L step

  multi_array<double,2> d_Lv({dvv_mult,r},stloc::device);
  multi_array<double,2> d_Lw({dvv_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Lvhat({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Lwhat({dvvh_mult,r},stloc::device);

  // Temporary to perform multiplications
  multi_array<double,2> d_tmpX({dxx_mult,r},stloc::device);
  multi_array<double,2> d_tmpS({r,r},stloc::device);
  multi_array<double,2> d_tmpS1({r,r},stloc::device);
  multi_array<double,2> d_tmpS2({r,r},stloc::device);
  multi_array<double,2> d_tmpS3({r,r},stloc::device);
  multi_array<double,2> d_tmpS4({r,r},stloc::device);
  multi_array<double,2> d_tmpV({dvv_mult,r}, stloc::device);

  multi_array<cuDoubleComplex,2> d_tmpXhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_tmpVhat({dvvh_mult,r},stloc::device);

  // For random values generation
  curandGenerator_t gen;

  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,time(0));

  // Quantities of interest

  double* d_el_energy_x;
  cudaMalloc((void**)&d_el_energy_x,sizeof(double));
  double* d_el_energy_y;
  cudaMalloc((void**)&d_el_energy_y,sizeof(double));

  double d_el_energy_CPU;

  multi_array<double,1> d_int_x({r},stloc::device);
  multi_array<double,1> d_int_v({r},stloc::device);

  double* d_mass;
  cudaMalloc((void**)&d_mass,sizeof(double));
  double d_mass_CPU;
  double err_mass_CPU;

  multi_array<double,1> d_we_v2({dvv_mult},stloc::device);
  multi_array<double,1> d_we_w2({dvv_mult},stloc::device);
  d_we_v2 = we_v2;
  d_we_w2 = we_w2;

  multi_array<double,1> d_int_v2({r},stloc::device);

  double* d_energy;
  cudaMalloc((void**)&d_energy,sizeof(double));
  double d_energy_CPU;
  double err_energy_CPU;

  ofstream el_energyGPUf;
  ofstream err_massGPUf;
  ofstream err_energyGPUf;

  el_energyGPUf.open("../../plots/el_energy_gpu_order1_2d.txt");
  err_massGPUf.open("../../plots/err_mass_gpu_order1_2d.txt");
  err_energyGPUf.open("../../plots/err_energy_gpu_order1_2d.txt");

  el_energyGPUf.precision(16);
  err_massGPUf.precision(16);
  err_energyGPUf.precision(16);

  el_energyGPUf << tstar << endl;
  el_energyGPUf << tau << endl;

  #endif

  // Main cycle
  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i + 1 << " on " << nsteps << endl;

    // CPU

    #ifdef __CPU__

    gt::start("Main loop CPU");

    /* K step */

    tmpX = lr_sol.X;

    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field

    integrate(lr_sol.V,-h_vv[0]*h_vv[1],rho);

    matvec(lr_sol.X,rho,ef);

    ef += 1.0;

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    #ifdef __OPENMP__
    #pragma omp parallel for
    for(Index j = 0; j < N_xx[1]; j++){
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
        complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

        efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
        efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
      }
    }
    #else
    for(Index j = 0; j < N_xx[1]; j++){
      if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
        complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

        efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
        efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
      }
    }
    #endif

    efhatx(0) = complex<double>(0.0,0.0);
    efhaty(0) = complex<double>(0.0,0.0);
    efhatx((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
    efhaty((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());

    // Main of K step

    coeff(lr_sol.V, lr_sol.V, we_v, C1v);
    coeff(lr_sol.V, lr_sol.V, we_w, C1w);

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

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
            complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
            Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_split*lambdax*dcv_r(k))*ncxx;
          }
        }
      }


      matmul_transb(Mhat,Tvc,Khat);

      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

      // Full step --
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

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_ee*lambday*dcw_r(k));
              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee*phi1_im(-tau_ee*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_ee*lambday*dcw_r(k));
              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee*phi1_im(-tau_ee*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Mhat,Twc,Khat);

        Khat *= ncxx;

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

        // Second stage

        ptw_mult_row(lr_sol.X,efx.begin(),Kex);
        ptw_mult_row(lr_sol.X,efy.begin(),Key);

        fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
        fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());

        matmul_transb(Kexhat,C2vc,Khat);
        matmul_transb(Keyhat,C2wc,Kexhat);

        Kexhat += Khat;
        matmul(Kexhat,Twc,Khat);

        Khat -= tmpXhat;

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee*phi2_im(-tau_ee*lambday*dcw_r(k))*Khat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee*phi2_im(-tau_ee*lambday*dcw_r(k))*Khat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Mhat,Twc,Khat);

        Khat *= ncxx;

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());


      }
    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_xx);

    /* S Step */
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index j = 0; j < (dxx_mult); j++){
      we_x(j) = efx(j) * h_xx[0] * h_xx[1];
      we_y(j) = efy(j) * h_xx[0] * h_xx[1];
    }


    coeff(lr_sol.X, lr_sol.X, we_x, D1x);
    coeff(lr_sol.X, lr_sol.X, we_y, D1y);

    fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)tmpXhat.begin());

    ptw_mult_row(tmpXhat,lambdax_n.begin(),dXhat_x);
    ptw_mult_row(tmpXhat,lambday_n.begin(),dXhat_y);

    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_x.begin(),dX_x.begin());

    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_y.begin(),dX_y.begin());

    coeff(lr_sol.X, dX_x, h_xx[0]*h_xx[1], D2x);
    coeff(lr_sol.X, dX_y, h_xx[0]*h_xx[1], D2y);

    // Runge Kutta 4
    for(Index jj = 0; jj< nsteps_rk4; jj++){
      matmul_transb(lr_sol.S,C1v,tmpS);
      matmul(D2x,tmpS,tmpS1);
      matmul_transb(lr_sol.S,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS1 += Tw;
      matmul_transb(lr_sol.S,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS1 -= Tw;
      matmul_transb(lr_sol.S,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS1 -= Tw;

      Tv = tmpS1;
      Tv *= (tau_rk4/2);
      Tv += lr_sol.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS2);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS2 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS2 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS2 -= Tw;

      Tv = tmpS2;
      Tv *= (tau_rk4/2);
      Tv += lr_sol.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS3);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS3 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS3 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS3 -= Tw;

      Tv = tmpS3;
      Tv *= tau_rk4;
      Tv += lr_sol.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS4);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS4 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS4 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS4 -= Tw;

      tmpS2 *= 2.0;
      tmpS3 *= 2.0;

      tmpS1 += tmpS2;
      tmpS1 += tmpS3;
      tmpS1 += tmpS4;
      tmpS1 *= (tau_rk4/6.0);

      lr_sol.S += tmpS1;

    }

    /* L step */
    tmpV = lr_sol.V;

    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    schur(D1x, Tx, dd1x_r, lwork);
    schur(D1y, Ty, dd1y_r, lwork);

    Tx.to_cplx(Txc);
    Ty.to_cplx(Tyc);
    D2x.to_cplx(D2xc);
    D2y.to_cplx(D2yc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

      matmul(Lhat,Txc,Nhat);
      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_vv[1]; j++){
          for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
            complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
            Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_split*lambdav*dd1x_r(k))*ncvv;
          }
        }
      }

      matmul_transb(Nhat,Txc,Lhat);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      // Full step --
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

      matmul(Lhat,Tyc,Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.V,v.begin(),Lv);
        ptw_mult_row(lr_sol.V,w.begin(),Lw);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());

        matmul_transb(Lvhat,D2xc,Lhat);
        matmul_transb(Lwhat,D2yc,tmpVhat);

        Lhat +=  tmpVhat;

        matmul(Lhat,Tyc,tmpVhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);
              Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_ee*lambdaw*dd1y_r(k));
              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee*phi1_im(tau_ee*lambdaw*dd1y_r(k))*tmpVhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);
              Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_ee*lambdaw*dd1y_r(k));
              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee*phi1_im(tau_ee*lambdaw*dd1y_r(k))*tmpVhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #endif


        matmul_transb(Nhat,Tyc,Lhat);

        Lhat *= ncvv;

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

        // Second stage

        ptw_mult_row(lr_sol.V,v.begin(),Lv);
        ptw_mult_row(lr_sol.V,w.begin(),Lw);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());

        matmul_transb(Lvhat,D2xc,Lhat);
        matmul_transb(Lwhat,D2yc,Lvhat);

        Lvhat += Lhat;
        matmul(Lvhat,Tyc,Lhat);

        Lhat -= tmpVhat;

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);

              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee*phi2_im(tau_ee*lambdaw*dd1y_r(k))*Lhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);

              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee*phi2_im(tau_ee*lambdaw*dd1y_r(k))*Lhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Nhat,Tyc,Lhat);

        Lhat *= ncvv;

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      }
    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_vv);

    transpose_inplace(lr_sol.S);

    gt::stop("Main loop CPU");

    // Electric energy
    el_energy = 0.0;
    #ifdef __OPENMP__
    #pragma omp parallel for reduction(+:el_energy)
    #endif
    for(Index ii = 0; ii < (dxx_mult); ii++){
      el_energy += 0.5*(pow(efx(ii),2)+pow(efy(ii),2))*h_xx[0]*h_xx[1];
    }
    el_energyf << el_energy << endl;

    // Error Mass
    integrate(lr_sol.X,h_xx[0]*h_xx[1],int_x);

    integrate(lr_sol.V,h_vv[0]*h_vv[1],int_v);

    matvec(lr_sol.S,int_v,rho);

    mass = 0.0;
    for(int ii = 0; ii < r; ii++){
      mass += (int_x(ii)*rho(ii));
    }

    err_mass = abs(mass0-mass);

    err_massf << err_mass << endl;

    // Error energy

    integrate(lr_sol.V,we_v2,int_v);

    integrate(lr_sol.V,we_w2,int_v2);

    int_v += int_v2;

    matvec(lr_sol.S,int_v,rho);

    energy = el_energy;
    for(int ii = 0; ii < r; ii++){
      energy += 0.5*(int_x(ii)*rho(ii));
    }

    err_energy = abs(energy0-energy);

    err_energyf << err_energy << endl;

    #endif

    #ifdef __CUDACC__

    /* K step */

    gt::start("Main loop GPU");

    d_tmpX = d_lr_sol.X;

    matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    integrate(d_lr_sol.V,-h_vv[0]*h_vv[1],d_rho);

    matvec(d_lr_sol.X,d_rho,d_ef);

    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier_2d<<<(dxxh_mult+n_threads-1)/n_threads,n_threads>>>(dxxh_mult, N_xx[0]/2+1, N_xx[1], d_efhat.begin(), d_lim_xx, ncxx, d_efhatx.begin(),d_efhaty.begin());

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatx.begin(),d_efx.begin());
    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhaty.begin(),d_efy.begin());

    coeff(d_lr_sol.V, d_lr_sol.V, d_we_v, d_C1v);
    coeff(d_lr_sol.V, d_lr_sol.V, d_we_w, d_C1w);

    cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_tmpVhat.begin());

    ptw_mult_row_cplx_fourier_2d<<<(dvvh_mult*r+n_threads-1)/n_threads,n_threads>>>(dvvh_mult*r, N_vv[0]/2+1, N_vv[1], d_tmpVhat.begin(), d_lim_vv, ncvv, d_dVhat_v.begin(), d_dVhat_w.begin()); // Very similar, maybe can be put together

    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_v.begin(),d_dV_v.begin());
    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_w.begin(),d_dV_w.begin());

    coeff(d_lr_sol.V, d_dV_v, h_vv[0]*h_vv[1], d_C2v);
    coeff(d_lr_sol.V, d_dV_w, h_vv[0]*h_vv[1], d_C2w);

    C1v_gpu = d_C1v;
    schur(C1v_gpu, Tv_gpu, dcv_r_gpu, lwork);
    d_Tv = Tv_gpu;
    d_dcv_r = dcv_r_gpu;

    C1w_gpu = d_C1w;
    schur(C1w_gpu, Tw_gpu, dcw_r_gpu, lwork);
    d_Tw = Tw_gpu;
    d_dcw_r = dcw_r_gpu;

    cplx_conv<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(), d_Tv.begin(), d_Tvc.begin());
    cplx_conv<<<(d_Tw.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tw.num_elements(), d_Tw.begin(), d_Twc.begin());
    cplx_conv<<<(d_C2v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2v.num_elements(), d_C2v.begin(), d_C2vc.begin());
    cplx_conv<<<(d_C2w.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2w.num_elements(), d_C2w.begin(), d_C2wc.begin());

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution

      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

      matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(), d_dcv_r.begin(), tau_split, d_lim_xx, ncxx);

      matmul_transb(d_Mhat,d_Tvc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      // Full step -- Exponential Euler
      //#ifdef __FFTW__
      //ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);
      //#else
      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin()); // check if CUDA preserves input and eventually adapt. YES
      //#endif

      matmul(d_Khat,d_Twc,d_Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());

        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());

        matmul_transb(d_Kexhat,d_C2vc,d_Khat);

        matmul_transb(d_Keyhat,d_C2wc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        matmul(d_Khat,d_Twc,d_tmpXhat);

        exp_euler_fourier_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(),d_dcw_r.begin(),tau_ee, d_lim_xx, d_tmpXhat.begin());

        matmul_transb(d_Mhat,d_Twc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());

        matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        matmul_transb(d_Keyhat,d_C2wc,d_Kexhat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        matmul(d_Kexhat,d_Twc,d_Khat);

        second_ord_stage_fourier_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(),d_dcw_r.begin(),tau_ee, d_lim_xx, d_tmpXhat.begin(), d_Khat.begin()); // Very similar, maybe can be put together

        matmul_transb(d_Mhat,d_Twc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      }
    }

    gram_schmidt_gpu(d_lr_sol.X, d_lr_sol.S, h_xx[0]*h_xx[1], gen);

    ptw_mult_scal<<<(d_efx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efx.num_elements(), d_efx.begin(), h_xx[0] * h_xx[1], d_we_x.begin());
    ptw_mult_scal<<<(d_efy.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efy.num_elements(), d_efy.begin(), h_xx[0] * h_xx[1], d_we_y.begin());

    coeff(d_lr_sol.X, d_lr_sol.X, d_we_x, d_D1x);

    coeff(d_lr_sol.X, d_lr_sol.X, d_we_y, d_D1y);

    cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_tmpXhat.begin()); // check if CUDA preserves input and eventually adapt

    ptw_mult_row_cplx_fourier_2d<<<(dxxh_mult*r+n_threads-1)/n_threads,n_threads>>>(dxxh_mult*r, N_xx[0]/2+1, N_xx[1], d_tmpXhat.begin(), d_lim_xx, ncxx, d_dXhat_x.begin(), d_dXhat_y.begin()); // Very similar, maybe can be put together

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_x.begin(),d_dX_x.begin());

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_y.begin(),d_dX_y.begin());

    coeff(d_lr_sol.X, d_dX_x, h_xx[0]*h_xx[1], d_D2x);

    coeff(d_lr_sol.X, d_dX_y, h_xx[0]*h_xx[1], d_D2y);

    // RK4
    for(Index jj = 0; jj< nsteps_rk4; jj++){
      matmul_transb(d_lr_sol.S,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS1);
      matmul_transb(d_lr_sol.S,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      matmul_transb(d_lr_sol.S,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      matmul_transb(d_lr_sol.S,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(), d_Tv.begin(), tau_rk4/2.0, d_lr_sol.S.begin(), d_tmpS1.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS2);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(), d_Tv.begin(), tau_rk4/2.0, d_lr_sol.S.begin(), d_tmpS2.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS3);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(), d_Tv.begin(), tau_rk4, d_lr_sol.S.begin(), d_tmpS3.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS4);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4_finalcomb<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(), d_lr_sol.S.begin(), tau_rk4, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_Tw.begin());

    }

    d_tmpV = d_lr_sol.V;

    matmul_transb(d_tmpV,d_lr_sol.S,d_lr_sol.V);

    D1x_gpu = d_D1x;
    schur(D1x_gpu, Tx_gpu, dd1x_r_gpu, lwork);
    d_Tx = Tx_gpu;
    d_dd1x_r = dd1x_r_gpu;

    D1y_gpu = d_D1y;
    schur(D1y_gpu, Ty_gpu, dd1y_r_gpu, lwork);
    d_Ty = Ty_gpu;
    d_dd1y_r = dd1y_r_gpu;

    cplx_conv<<<(d_Tx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tx.num_elements(), d_Tx.begin(), d_Txc.begin());
    cplx_conv<<<(d_Ty.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Ty.num_elements(), d_Ty.begin(), d_Tyc.begin());
    cplx_conv<<<(d_D2x.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2x.num_elements(), d_D2x.begin(), d_D2xc.begin());
    cplx_conv<<<(d_D2y.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2y.num_elements(), d_D2y.begin(), d_D2yc.begin());

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution
      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

      matmul(d_Lhat,d_Txc,d_Nhat);

      exact_sol_exp_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(), d_dd1x_r.begin(), -tau_split, d_lim_vv, ncvv);

      matmul_transb(d_Nhat,d_Txc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      // Full step --
      //#ifdef __FFTW__
      //ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), 1.0/ncvv);
      //#else
      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin()); // check if CUDA preserves input and eventually adapt
      //#endif

      matmul(d_Lhat,d_Tyc,d_Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin()); // check if CUDA preserves input and eventually adapt
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin()); // check if CUDA preserves input and eventually adapt

        matmul_transb(d_Lvhat,d_D2xc,d_Lhat);
        matmul_transb(d_Lwhat,d_D2yc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        matmul(d_Lhat,d_Tyc,d_tmpVhat);

        exp_euler_fourier_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(),d_dd1y_r.begin(),-tau_ee, d_lim_vv, d_tmpVhat.begin());

        matmul_transb(d_Nhat,d_Tyc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin()); // check if CUDA preserves input and eventually adapt
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin()); // check if CUDA preserves input and eventually adapt

        matmul_transb(d_Lvhat,d_D2xc,d_Lhat);
        matmul_transb(d_Lwhat,d_D2yc,d_Lvhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lvhat.begin(), d_Lhat.begin());

        matmul(d_Lvhat,d_Tyc,d_Lhat);

        second_ord_stage_fourier_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(),d_dd1y_r.begin(), -tau_ee, d_lim_vv, d_tmpVhat.begin(), d_Lhat.begin());

        matmul_transb(d_Nhat,d_Tyc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());


      }
    }

    gram_schmidt_gpu(d_lr_sol.V, d_lr_sol.S, h_vv[0]*h_vv[1], gen);

    transpose_inplace<<<d_lr_sol.S.num_elements(),1>>>(r,d_lr_sol.S.begin());

    cudaDeviceSynchronize();
    gt::stop("Main loop GPU");

    // Electric energy

    cublasDdot (handle_dot, d_efx.num_elements(), d_efx.begin(), 1, d_efx.begin(), 1, d_el_energy_x);
    cublasDdot (handle_dot, d_efy.num_elements(), d_efy.begin(), 1, d_efy.begin(), 1, d_el_energy_y);
    cudaDeviceSynchronize(); // as handle_dot is asynchronous
    ptw_sum<<<1,1>>>(1,d_el_energy_x,d_el_energy_y); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
    scale_unique<<<1,1>>>(d_el_energy_x,0.5*h_xx[0]*h_xx[1]); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_el_energy_CPU,d_el_energy_x,sizeof(double),cudaMemcpyDeviceToHost);

    el_energyGPUf << d_el_energy_CPU << endl;

    // Error in mass

    integrate(d_lr_sol.X,h_xx[0]*h_xx[1],d_int_x);

    integrate(d_lr_sol.V,h_vv[0]*h_vv[1],d_int_v);

    matvec(d_lr_sol.S,d_int_v,d_rho);

    cublasDdot (handle_dot, r, d_int_x.begin(), 1, d_rho.begin(), 1,d_mass);
    cudaDeviceSynchronize();

    cudaMemcpy(&d_mass_CPU,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    err_mass_CPU = abs(mass0-d_mass_CPU);
    err_massGPUf << err_mass_CPU << endl;

    // Error in energy
    integrate(d_lr_sol.V,d_we_v2,d_int_v);

    integrate(d_lr_sol.V,d_we_w2,d_int_v2);

    ptw_sum<<<(d_int_v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_int_v.num_elements(),d_int_v.begin(),d_int_v2.begin());


    matvec(d_lr_sol.S,d_int_v,d_rho);


    cublasDdot (handle_dot, r, d_int_x.begin(), 1, d_rho.begin(), 1, d_energy);
    cudaDeviceSynchronize();
     scale_unique<<<1,1>>>(d_energy,0.5); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_energy_CPU,d_energy,sizeof(double),cudaMemcpyDeviceToHost);

    err_energy_CPU = abs(energy0-(d_energy_CPU+d_el_energy_CPU));
    err_energyGPUf << err_energy_CPU << endl;

    #endif
    #ifdef __CPU__
    cout << "Electric energy: " << el_energy << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Electric energy GPU: " << d_el_energy_CPU << endl;
    #endif
    #ifdef __CPU__
    cout << "Error in mass: " << err_mass << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Error in mass GPU: " << err_mass_CPU << endl;
    #endif
    #ifdef __CPU__
    cout << "Error in energy: " << err_energy << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Error in energy GPU: " << err_energy_CPU << endl;
    #endif

  }
  #ifdef __CPU__
  el_energyf.close();
  err_massf.close();
  err_energyf.close();
  #endif
  #ifdef __CUDACC__
  cublasDestroy(handle);
  cublasDestroy(handle_dot);

  destroy_plans(plans_d_e);
  destroy_plans(d_plans_xx);
  destroy_plans(d_plans_vv);

  curandDestroyGenerator(gen);

  el_energyGPUf.close();
  err_massGPUf.close();
  err_energyGPUf.close();
  #endif

  #ifdef __CPU__
  return lr_sol;
  #else
  lr_sol.X = d_lr_sol.X;
  lr_sol.S = d_lr_sol.S;
  lr_sol.V = d_lr_sol.V;
  cudaDeviceSynchronize();
  return lr_sol;
  #endif
}

lr2<double> integration_second_order(array<Index,2> N_xx,array<Index,2> N_vv, int r,double tstar, Index nsteps, int nsteps_split, int nsteps_ee, int nsteps_rk4, array<double,4> lim_xx, array<double,4> lim_vv, double alpha, double kappa1, double kappa2, lr2<double> lr_sol, array<fftw_plan,2> plans_e, array<fftw_plan,2> plans_xx, array<fftw_plan,2> plans_vv){

  double tau = tstar/nsteps;
  double tau_h = tau/2.0;

  double tau_split = tau/nsteps_split;
  double tau_split_h = tau_h/nsteps_split;

  double tau_ee = tau_split / nsteps_ee;
  double tau_ee_h = tau_split_h / nsteps_ee;

  double tau_rk4_h = tau_h / nsteps_rk4;

  array<double,2> h_xx, h_vv;
  int jj = 0;
  for(int ii = 0; ii < 2; ii++){
    h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
    h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
    jj+=2;
  }

  Index dxx_mult = N_xx[0]*N_xx[1];
  Index dxxh_mult = N_xx[1]*(N_xx[0]/2 + 1);

  Index dvv_mult = N_vv[0]*N_vv[1];
  Index dvvh_mult = N_vv[1]*(N_vv[0]/2 + 1);

  multi_array<double,1> v({dvv_mult});
  multi_array<double,1> w({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < N_vv[1]; j++){
    for(Index i = 0; i < N_vv[0]; i++){
      v(i+j*N_vv[0]) = lim_vv[0] + i*h_vv[0];
      w(i+j*N_vv[0]) = lim_vv[2] + j*h_vv[1];
    }
  }

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({dxx_mult});
  multi_array<double,1> efx({dxx_mult});
  multi_array<double,1> efy({dxx_mult});

  multi_array<complex<double>,1> efhat({dxxh_mult});
  multi_array<complex<double>,1> efhatx({dxxh_mult});
  multi_array<complex<double>,1> efhaty({dxxh_mult});

  // Some FFT stuff for X and V

  multi_array<complex<double>,1> lambdax_n({dxxh_mult});
  multi_array<complex<double>,1> lambday_n({dxxh_mult});

  double ncxx = 1.0 / (dxx_mult);

  Index mult;
  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index j = 0; j < N_xx[1]; j++){
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
      lambdax_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i)*ncxx;
      lambday_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult)*ncxx;
    }
  }
  #else
  for(Index j = 0; j < N_xx[1]; j++){
    if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      lambdax_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i)*ncxx;
      lambday_n(i+j*(N_xx[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult)*ncxx;
    }
  }
  #endif
  multi_array<complex<double>,1> lambdav_n({dvvh_mult});
  multi_array<complex<double>,1> lambdaw_n({dvvh_mult});

  double ncvv = 1.0 / (dvv_mult);

  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index j = 0; j < N_vv[1]; j++){
    for(Index i = 0; i < (N_vv[0]/2+1); i++){
      if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
      lambdav_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i)*ncvv;
      lambdaw_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult)*ncvv;
    }
  }
  #else
  for(Index j = 0; j < N_vv[1]; j++){
    if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
    for(Index i = 0; i < (N_vv[0]/2+1); i++){
      lambdav_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i)*ncvv;
      lambdaw_n(i+j*(N_vv[0]/2+1)) = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult)*ncvv;
    }
  }
  #endif

  multi_array<complex<double>,2> Khat({dxxh_mult,r});

  multi_array<complex<double>,2> Lhat({dvvh_mult,r});

  // For C coefficients
  multi_array<double,2> C1v({r,r});
  multi_array<double,2> C1w({r,r});

  multi_array<double,2> C2v({r,r});
  multi_array<double,2> C2w({r,r});

  multi_array<complex<double>,2> C2vc({r,r});
  multi_array<complex<double>,2> C2wc({r,r});

  multi_array<double,1> we_v({dvv_mult});
  multi_array<double,1> we_w({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
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

  multi_array<complex<double>,2> D2xc({r,r});
  multi_array<complex<double>,2> D2yc({r,r});

  multi_array<double,1> we_x({dxx_mult});
  multi_array<double,1> we_y({dxx_mult});

  multi_array<double,2> dX_x({dxx_mult,r});
  multi_array<double,2> dX_y({dxx_mult,r});

  multi_array<complex<double>,2> dXhat_x({dxxh_mult,r});
  multi_array<complex<double>,2> dXhat_y({dxxh_mult,r});

  // For Schur decomposition
  multi_array<double,1> dcv_r({r});
  multi_array<double,1> dcw_r({r});
  multi_array<double,1> dd1x_r({r});
  multi_array<double,1> dd1y_r({r});

  multi_array<double,2> Tv({r,r});
  multi_array<double,2> Tw({r,r});
  multi_array<double,2> Tx({r,r});
  multi_array<double,2> Ty({r,r});

  multi_array<complex<double>,2> Mhat({dxxh_mult,r});
  multi_array<complex<double>,2> Nhat({dvvh_mult,r});
  multi_array<complex<double>,2> Tvc({r,r});
  multi_array<complex<double>,2> Twc({r,r});
  multi_array<complex<double>,2> Txc({r,r});
  multi_array<complex<double>,2> Tyc({r,r});

  #ifdef __MKL__
    MKL_INT lwork = -1;
  #else
    int lwork = -1;
  #endif

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
  multi_array<double,2> tmpS1({r,r});
  multi_array<double,2> tmpS2({r,r});
  multi_array<double,2> tmpS3({r,r});
  multi_array<double,2> tmpS4({r,r});

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

  // Initial mass
  integrate(lr_sol.X,h_xx[0]*h_xx[1],int_x);
  integrate(lr_sol.V,h_vv[0]*h_vv[1],int_v);

  matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    mass0 += (int_x(ii)*rho(ii));
  }

  // Initial energy
  integrate(lr_sol.V,h_vv[0]*h_vv[1],rho);
  rho *= -1.0;
  matmul(lr_sol.X,lr_sol.S,tmpX);
  matvec(tmpX,rho,ef);
  ef += 1.0;
  fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());
  #ifdef __OPENMP__
  #pragma omp parallel for
  for(Index j = 0; j < N_xx[1]; j++){
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);
      efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
      efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
    }
  }
  #else
  for(Index j = 0; j < N_xx[1]; j++){
    if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
    for(Index i = 0; i < (N_xx[0]/2+1); i++){
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
      complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);
      efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
      efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
    }
  }
  #endif

  efhatx(0) = complex<double>(0.0,0.0);
  efhaty(0) = complex<double>(0.0,0.0);
  efhatx((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
  efhaty((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());

  energy0 = 0.0;
  #ifdef __OPENMP__
  #pragma omp parallel for reduction(+:energy0)
  #endif
  for(Index ii = 0; ii < (dxx_mult); ii++){
    energy0 += 0.5*(pow(efx(ii),2)+pow(efy(ii),2))*h_xx[0]*h_xx[1];
  }

  multi_array<double,1> we_v2({dvv_mult});
  multi_array<double,1> we_w2({dvv_mult});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < (dvv_mult); j++){
    we_v2(j) = pow(v(j),2) * h_vv[0] * h_vv[1];
    we_w2(j) = pow(w(j),2) * h_vv[0] * h_vv[1];
  }

  integrate(lr_sol.V,we_v2,int_v);
  integrate(lr_sol.V,we_w2,int_v2);

  int_v += int_v2;

  matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    energy0 += 0.5*(int_x(ii)*rho(ii));
  }

  #ifdef __CPU__
  ofstream el_energyf;
  ofstream err_massf;
  ofstream err_energyf;

  el_energyf.open("../../plots/el_energy_order2_2d.txt");
  err_massf.open("../../plots/err_mass_order2_2d.txt");
  err_energyf.open("../../plots/err_energy_order2_2d.txt");

  el_energyf.precision(16);
  err_massf.precision(16);
  err_energyf.precision(16);

  el_energyf << tstar << endl;
  el_energyf << tau << endl;
  #endif
  // Additional stuff for second order
  lr2<double> lr_sol_e(r,{dxx_mult,dvv_mult});

  //// FOR GPU ////

  #ifdef __CUDACC__
  cublasCreate (&handle);
  cublasCreate (&handle_dot);
  cublasSetPointerMode(handle_dot, CUBLAS_POINTER_MODE_DEVICE);

  // To be substituted if we initialize in GPU

  lr2<double> d_lr_sol(r,{dxx_mult,dvv_mult},stloc::device);
  d_lr_sol.X = lr_sol.X;
  d_lr_sol.V = lr_sol.V;
  d_lr_sol.S = lr_sol.S;

  // For Electric field

  multi_array<double,1> d_rho({r},stloc::device);
  multi_array<double,1> d_ef({dxx_mult},stloc::device);
  multi_array<double,1> d_efx({dxx_mult},stloc::device);
  multi_array<double,1> d_efy({dxx_mult},stloc::device);

  multi_array<cuDoubleComplex,1> d_efhat({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhatx({dxxh_mult},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhaty({dxxh_mult},stloc::device);

  array<cufftHandle,2> plans_d_e = create_plans_2d(N_xx, 1);

  // FFT stuff

  double* d_lim_xx;
  cudaMalloc((void**)&d_lim_xx,4*sizeof(double));
  cudaMemcpy(d_lim_xx,lim_xx.begin(),4*sizeof(double),cudaMemcpyHostToDevice);

  double* d_lim_vv;
  cudaMalloc((void**)&d_lim_vv,4*sizeof(double));
  cudaMemcpy(d_lim_vv,lim_vv.begin(),4*sizeof(double),cudaMemcpyHostToDevice);

  multi_array<cuDoubleComplex,2> d_Khat({dxxh_mult,r},stloc::device);
  array<cufftHandle,2> d_plans_xx = create_plans_2d(N_xx, r);

  multi_array<cuDoubleComplex,2> d_Lhat({dvvh_mult,r},stloc::device);
  array<cufftHandle,2> d_plans_vv = create_plans_2d(N_vv, r);

  // C coefficients

  multi_array<double,2> d_C1v({r,r},stloc::device);
  multi_array<double,2> d_C1w({r,r}, stloc::device);

  multi_array<double,2> d_C2v({r,r},stloc::device);
  multi_array<double,2> d_C2w({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_C2vc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2wc({r,r},stloc::device);

  multi_array<double,1> d_we_v({dvv_mult},stloc::device);
  d_we_v = we_v;

  multi_array<double,1> d_we_w({dvv_mult},stloc::device);
  d_we_w = we_w;

  multi_array<double,2> d_dV_v({dvv_mult,r},stloc::device);
  multi_array<double,2> d_dV_w({dvv_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_dVhat_v({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dVhat_w({dvvh_mult,r},stloc::device);


  // D coefficients

  multi_array<double,2> d_D1x({r,r},stloc::device);
  multi_array<double,2> d_D1y({r,r},stloc::device);

  multi_array<double,2> d_D2x({r,r},stloc::device);
  multi_array<double,2> d_D2y({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_D2xc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_D2yc({r,r},stloc::device);


  multi_array<double,1> d_we_x({dxx_mult},stloc::device);
  multi_array<double,1> d_we_y({dxx_mult},stloc::device);

  multi_array<double,2> d_dX_x({dxx_mult,r},stloc::device);
  multi_array<double,2> d_dX_y({dxx_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_dXhat_x({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_dXhat_y({dxxh_mult,r},stloc::device);

  multi_array<double,1> d_v({dvv_mult},stloc::device);
  d_v = v;
  multi_array<double,1> d_w({dvv_mult},stloc::device);
  d_w = w;


  // Schur decomposition

  multi_array<double,2> C1v_gpu({r,r});
  multi_array<double,2> Tv_gpu({r,r});
  multi_array<double,1> dcv_r_gpu({r});
  multi_array<double,2> C1w_gpu({r,r});
  multi_array<double,2> Tw_gpu({r,r});
  multi_array<double,1> dcw_r_gpu({r});

  multi_array<double,2> D1x_gpu({r,r});
  multi_array<double,2> Tx_gpu({r,r});
  multi_array<double,1> dd1x_r_gpu({r});
  multi_array<double,2> D1y_gpu({r,r});
  multi_array<double,2> Ty_gpu({r,r});
  multi_array<double,1> dd1y_r_gpu({r});

  multi_array<double,1> d_dcv_r({r},stloc::device);
  multi_array<double,2> d_Tv({r,r},stloc::device);
  multi_array<double,1> d_dcw_r({r},stloc::device);
  multi_array<double,2> d_Tw({r,r},stloc::device);

  multi_array<double,1> d_dd1x_r({r},stloc::device);
  multi_array<double,2> d_Tx({r,r},stloc::device);
  multi_array<double,1> d_dd1y_r({r},stloc::device);
  multi_array<double,2> d_Ty({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Mhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Nhat({dvvh_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Tvc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Twc({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Txc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Tyc({r,r},stloc::device);

  // For K step

  multi_array<double,2> d_Kex({dxx_mult,r},stloc::device);
  multi_array<double,2> d_Key({dxx_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Kexhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Keyhat({dxxh_mult,r},stloc::device);

  // For L step

  multi_array<double,2> d_Lv({dvv_mult,r},stloc::device);
  multi_array<double,2> d_Lw({dvv_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Lvhat({dvvh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Lwhat({dvvh_mult,r},stloc::device);

  // Temporary to perform multiplications
  multi_array<double,2> d_tmpX({dxx_mult,r},stloc::device);
  multi_array<double,2> d_tmpS({r,r},stloc::device);
  multi_array<double,2> d_tmpS1({r,r},stloc::device);
  multi_array<double,2> d_tmpS2({r,r},stloc::device);
  multi_array<double,2> d_tmpS3({r,r},stloc::device);
  multi_array<double,2> d_tmpS4({r,r},stloc::device);
  multi_array<double,2> d_tmpV({dvv_mult,r}, stloc::device);

  multi_array<cuDoubleComplex,2> d_tmpXhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_tmpVhat({dvvh_mult,r},stloc::device);

  // For random values generation
  curandGenerator_t gen;

  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,time(0));

  // Quantities of interest

  double* d_el_energy_x;
  cudaMalloc((void**)&d_el_energy_x,sizeof(double));
  double* d_el_energy_y;
  cudaMalloc((void**)&d_el_energy_y,sizeof(double));

  double d_el_energy_CPU;

  multi_array<double,1> d_int_x({r},stloc::device);
  multi_array<double,1> d_int_v({r},stloc::device);

  double* d_mass;
  cudaMalloc((void**)&d_mass,sizeof(double));
  double d_mass_CPU;
  double err_mass_CPU;

  multi_array<double,1> d_we_v2({dvv_mult},stloc::device);
  multi_array<double,1> d_we_w2({dvv_mult},stloc::device);
  d_we_v2 = we_v2;
  d_we_w2 = we_w2;

  multi_array<double,1> d_int_v2({r},stloc::device);

  double* d_energy;
  cudaMalloc((void**)&d_energy,sizeof(double));
  double d_energy_CPU;
  double err_energy_CPU;

  ofstream el_energyGPUf;
  ofstream err_massGPUf;
  ofstream err_energyGPUf;

  el_energyGPUf.open("../../plots/el_energy_gpu_order2_2d.txt");
  err_massGPUf.open("../../plots/err_mass_gpu_order2_2d.txt");
  err_energyGPUf.open("../../plots/err_energy_gpu_order2_2d.txt");

  el_energyGPUf.precision(16);
  err_massGPUf.precision(16);
  err_energyGPUf.precision(16);

  el_energyGPUf << tstar << endl;
  el_energyGPUf << tau << endl;

  // Additional stuff for second order
  lr2<double> d_lr_sol_e(r,{dxx_mult,dvv_mult}, stloc::device);

  #endif

  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i + 1 << " on " << nsteps << endl;

    #ifdef __CPU__
    /* Lie splitting to obtain the electric field */

    lr_sol_e.X = lr_sol.X;
    lr_sol_e.S = lr_sol.S;
    lr_sol_e.V = lr_sol.V;

    /* Full step K until tau/2 */

    gt::start("Lie splitting for electric field CPU");

    tmpX = lr_sol_e.X;

    matmul(tmpX,lr_sol_e.S,lr_sol_e.X);

    // Electric field

    integrate(lr_sol_e.V,-h_vv[0]*h_vv[1],rho);

    matvec(lr_sol_e.X,rho,ef);

    ef += 1.0;

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    #ifdef __OPENMP__
    #pragma omp parallel for
    for(Index j = 0; j < N_xx[1]; j++){
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
        complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

        efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
        efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
      }
    }
    #else
    for(Index j = 0; j < N_xx[1]; j++){
      if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
        complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

        efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
        efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
      }
    }
    #endif

    efhatx(0) = complex<double>(0.0,0.0);
    efhaty(0) = complex<double>(0.0,0.0);
    efhatx((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
    efhaty((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());

    // Electric energy
    el_energy = 0.0;
    #ifdef __OPENMP__
    #pragma omp parallel for reduction(+:el_energy)
    #endif
    for(Index ii = 0; ii < (dxx_mult); ii++){
      el_energy += 0.5*(pow(efx(ii),2)+pow(efy(ii),2))*h_xx[0]*h_xx[1];
    }
    el_energyf << el_energy << endl;

    // Main of K step

    coeff(lr_sol_e.V, lr_sol_e.V, we_v, C1v);
    coeff(lr_sol_e.V, lr_sol_e.V, we_w, C1w);

    fftw_execute_dft_r2c(plans_vv[0],lr_sol_e.V.begin(),(fftw_complex*)tmpVhat.begin());

    ptw_mult_row(tmpVhat,lambdav_n.begin(),dVhat_v);
    ptw_mult_row(tmpVhat,lambdaw_n.begin(),dVhat_w);

    fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_v.begin(),dV_v.begin());
    fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_w.begin(),dV_w.begin());

    coeff(lr_sol_e.V, dV_v, h_vv[0]*h_vv[1], C2v);
    coeff(lr_sol_e.V, dV_w, h_vv[0]*h_vv[1], C2w);

    schur(C1v, Tv, dcv_r, lwork);
    schur(C1w, Tw, dcw_r, lwork);

    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    C2v.to_cplx(C2vc);
    C2w.to_cplx(C2wc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution

      fftw_execute_dft_r2c(plans_xx[0],lr_sol_e.X.begin(),(fftw_complex*)Khat.begin());

      matmul(Khat,Tvc,Mhat);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
            complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
            Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_split_h*lambdax*dcv_r(k))*ncxx;
          }
        }
      }


      matmul_transb(Mhat,Tvc,Khat);

      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol_e.X.begin());

      // Full step --
      fftw_execute_dft_r2c(plans_xx[0],lr_sol_e.X.begin(),(fftw_complex*)Khat.begin());

      matmul(Khat,Twc,Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol_e.X,efx.begin(),Kex);
        ptw_mult_row(lr_sol_e.X,efy.begin(),Key);

        fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
        fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());

        matmul_transb(Kexhat,C2vc,Khat);
        matmul_transb(Keyhat,C2wc,tmpXhat);

        Khat += tmpXhat;

        matmul(Khat,Twc,tmpXhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_ee_h*lambday*dcw_r(k));
              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi1_im(-tau_ee_h*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_ee_h*lambday*dcw_r(k));
              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi1_im(-tau_ee_h*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Mhat,Twc,Khat);

        Khat *= ncxx;

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol_e.X.begin());

        // Second stage

        ptw_mult_row(lr_sol_e.X,efx.begin(),Kex);
        ptw_mult_row(lr_sol_e.X,efy.begin(),Key);

        fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
        fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());

        matmul_transb(Kexhat,C2vc,Khat);
        matmul_transb(Keyhat,C2wc,Kexhat);

        Kexhat += Khat;
        matmul(Kexhat,Twc,Khat);

        Khat -= tmpXhat;

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi2_im(-tau_ee_h*lambday*dcw_r(k))*Khat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi2_im(-tau_ee_h*lambday*dcw_r(k))*Khat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Mhat,Twc,Khat);

        Khat *= ncxx;

        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol_e.X.begin());


      }
    }

    gram_schmidt(lr_sol_e.X, lr_sol_e.S, ip_xx);

    /* Full step S until tau/2 */

    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index j = 0; j < (dxx_mult); j++){
      we_x(j) = efx(j) * h_xx[0] * h_xx[1];
      we_y(j) = efy(j) * h_xx[0] * h_xx[1];
    }

    coeff(lr_sol_e.X, lr_sol_e.X, we_x, D1x);
    coeff(lr_sol_e.X, lr_sol_e.X, we_y, D1y);

    fftw_execute_dft_r2c(plans_xx[0],lr_sol_e.X.begin(),(fftw_complex*)tmpXhat.begin());

    ptw_mult_row(tmpXhat,lambdax_n.begin(),dXhat_x);
    ptw_mult_row(tmpXhat,lambday_n.begin(),dXhat_y);

    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_x.begin(),dX_x.begin());

    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_y.begin(),dX_y.begin());

    coeff(lr_sol_e.X, dX_x, h_xx[0]*h_xx[1], D2x);
    coeff(lr_sol_e.X, dX_y, h_xx[0]*h_xx[1], D2y);

    // Runge Kutta 4
    for(Index jj = 0; jj< nsteps_rk4; jj++){
      matmul_transb(lr_sol_e.S,C1v,tmpS);
      matmul(D2x,tmpS,tmpS1);
      matmul_transb(lr_sol_e.S,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS1 += Tw;
      matmul_transb(lr_sol_e.S,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS1 -= Tw;
      matmul_transb(lr_sol_e.S,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS1 -= Tw;

      Tv = tmpS1;
      Tv *= (tau_rk4_h/2);
      Tv += lr_sol_e.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS2);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS2 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS2 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS2 -= Tw;

      Tv = tmpS2;
      Tv *= (tau_rk4_h/2);
      Tv += lr_sol_e.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS3);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS3 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS3 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS3 -= Tw;

      Tv = tmpS3;
      Tv *= tau_rk4_h;
      Tv += lr_sol_e.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS4);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS4 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS4 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS4 -= Tw;

      tmpS2 *= 2.0;
      tmpS3 *= 2.0;

      tmpS1 += tmpS2;
      tmpS1 += tmpS3;
      tmpS1 += tmpS4;
      tmpS1 *= (tau_rk4_h/6.0);

      lr_sol_e.S += tmpS1;

    }

    /* Full step L until tau/2 */

    tmpV = lr_sol_e.V;

    matmul_transb(tmpV,lr_sol_e.S,lr_sol_e.V);

    schur(D1x, Tx, dd1x_r, lwork);
    schur(D1y, Ty, dd1y_r, lwork);

    Tx.to_cplx(Txc);
    Ty.to_cplx(Tyc);
    D2x.to_cplx(D2xc);
    D2y.to_cplx(D2yc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution
      fftw_execute_dft_r2c(plans_vv[0],lr_sol_e.V.begin(),(fftw_complex*)Lhat.begin());

      matmul(Lhat,Txc,Nhat);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_vv[1]; j++){
          for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
            complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
            Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_split_h*lambdav*dd1x_r(k))*ncvv;
          }
        }
      }

      matmul_transb(Nhat,Txc,Lhat);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol_e.V.begin());

      // Full step --
      fftw_execute_dft_r2c(plans_vv[0],lr_sol_e.V.begin(),(fftw_complex*)Lhat.begin());

      matmul(Lhat,Tyc,Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol_e.V,v.begin(),Lv);
        ptw_mult_row(lr_sol_e.V,w.begin(),Lw);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());

        matmul_transb(Lvhat,D2xc,Lhat);
        matmul_transb(Lwhat,D2yc,tmpVhat);

        Lhat +=  tmpVhat;

        matmul(Lhat,Tyc,tmpVhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);
              Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_ee_h*lambdaw*dd1y_r(k));
              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee_h*phi1_im(tau_ee_h*lambdaw*dd1y_r(k))*tmpVhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);
              Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_ee_h*lambdaw*dd1y_r(k));
              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee_h*phi1_im(tau_ee_h*lambdaw*dd1y_r(k))*tmpVhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Nhat,Tyc,Lhat);

        Lhat *= ncvv;

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol_e.V.begin());

        // Second stage

        ptw_mult_row(lr_sol_e.V,v.begin(),Lv);
        ptw_mult_row(lr_sol_e.V,w.begin(),Lw);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());

        matmul_transb(Lvhat,D2xc,Lhat);
        matmul_transb(Lwhat,D2yc,Lvhat);

        Lvhat += Lhat;
        matmul(Lvhat,Tyc,Lhat);

        Lhat -= tmpVhat;

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);

              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee_h*phi2_im(tau_ee_h*lambdaw*dd1y_r(k))*Lhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);

              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee_h*phi2_im(tau_ee_h*lambdaw*dd1y_r(k))*Lhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Nhat,Tyc,Lhat);

        Lhat *= ncvv;

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol_e.V.begin());

      }
    }

    gt::stop("Lie splitting for electric field CPU");

    gt::start("Restarted integration CPU");

    // Electric field at time tau/2

    integrate(lr_sol_e.V,-h_vv[0]*h_vv[1],rho);
    matvec(lr_sol_e.X,rho,ef);

    ef += 1.0;

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    #ifdef __OPENMP__
    #pragma omp parallel for collapse(2)
    for(Index j = 0; j < N_xx[1]; j++){
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
        complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

        efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
        efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
      }
    }
    #else
    for(Index j = 0; j < N_xx[1]; j++){
      if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
      for(Index i = 0; i < (N_xx[0]/2+1); i++){
        complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
        complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

        efhatx(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambdax / (pow(lambdax,2) + pow(lambday,2)) * ncxx;
        efhaty(i+j*(N_xx[0]/2+1)) = efhat(i+j*(N_xx[0]/2+1)) * lambday / (pow(lambdax,2) + pow(lambday,2)) * ncxx ;
      }
    }
    #endif

    efhatx(0) = complex<double>(0.0,0.0);
    efhaty(0) = complex<double>(0.0,0.0);
    efhatx((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
    efhaty((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());

    // Here I have the electric field at time tau/2, so restart integration

    // Half step K (until tau/2)

    tmpX = lr_sol.X;
    matmul(tmpX,lr_sol.S,lr_sol.X);

    for(Index ii = 0; ii < nsteps_split; ii++){
      // Half step -- Exact solution

      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

      matmul(Khat,Tvc,Mhat);

      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
            complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
            Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_split_h/2.0*lambdax*dcv_r(k))*ncxx;
          }
        }
      }

      matmul_transb(Mhat,Tvc,Khat);

      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

      // Full step --
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

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_ee_h*lambday*dcw_r(k));
              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi1_im(-tau_ee_h*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_ee_h*lambday*dcw_r(k));
              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi1_im(-tau_ee_h*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Mhat,Twc,Khat);

        Khat *= ncxx;
        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

        // Second stage

        ptw_mult_row(lr_sol.X,efx.begin(),Kex);
        ptw_mult_row(lr_sol.X,efy.begin(),Key);

        fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
        fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());

        matmul_transb(Kexhat,C2vc,Khat);
        matmul_transb(Keyhat,C2wc,Kexhat);

        Kexhat += Khat;
        matmul(Kexhat,Twc,Khat);

        Khat -= tmpXhat;

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi2_im(-tau_ee_h*lambday*dcw_r(k))*Khat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi2_im(-tau_ee_h*lambday*dcw_r(k))*Khat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Mhat,Twc,Khat);

        if(jj != (nsteps_ee -1)){
          Khat *= ncxx;
          fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin()); // Could be avoided if we do more just one step of the integrator
        }
      }

      // Half step -- exact solution

      matmul(Khat,Tvc,Mhat);

      #ifdef __OPENMP__
      #pragma omp parallel for
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
            complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
            Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_split_h/2.0*lambdax*dcv_r(k))*ncxx;
          }
        }
      }

      matmul_transb(Mhat,Tvc,Khat);

      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());


    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_xx);


    // Half step S (until tau/2)
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index j = 0; j < (dxx_mult); j++){
      we_x(j) = efx(j) * h_xx[0] * h_xx[1];
      we_y(j) = efy(j) * h_xx[0] * h_xx[1];
    }


    coeff(lr_sol.X, lr_sol.X, we_x, D1x);
    coeff(lr_sol.X, lr_sol.X, we_y, D1y);

    fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)tmpXhat.begin());

    ptw_mult_row(tmpXhat,lambdax_n.begin(),dXhat_x);
    ptw_mult_row(tmpXhat,lambday_n.begin(),dXhat_y);

    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_x.begin(),dX_x.begin());

    fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)dXhat_y.begin(),dX_y.begin());

    coeff(lr_sol.X, dX_x, h_xx[0]*h_xx[1], D2x);
    coeff(lr_sol.X, dX_y, h_xx[0]*h_xx[1], D2y);

    // Runge Kutta 4
    for(Index jj = 0; jj< nsteps_rk4; jj++){
      matmul_transb(lr_sol.S,C1v,tmpS);
      matmul(D2x,tmpS,tmpS1);
      matmul_transb(lr_sol.S,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS1 += Tw;
      matmul_transb(lr_sol.S,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS1 -= Tw;
      matmul_transb(lr_sol.S,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS1 -= Tw;

      Tv = tmpS1;
      Tv *= (tau_rk4_h/2);
      Tv += lr_sol.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS2);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS2 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS2 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS2 -= Tw;

      Tv = tmpS2;
      Tv *= (tau_rk4_h/2);
      Tv += lr_sol.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS3);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS3 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS3 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS3 -= Tw;

      Tv = tmpS3;
      Tv *= tau_rk4_h;
      Tv += lr_sol.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS4);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS4 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS4 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS4 -= Tw;

      tmpS2 *= 2.0;
      tmpS3 *= 2.0;

      tmpS1 += tmpS2;
      tmpS1 += tmpS3;
      tmpS1 += tmpS4;
      tmpS1 *= (tau_rk4_h/6.0);

      lr_sol.S += tmpS1;

    }


    // Full step L (until tau)

    tmpV = lr_sol.V;


    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    schur(D1x, Tx, dd1x_r, lwork);
    schur(D1y, Ty, dd1y_r, lwork);

    Tx.to_cplx(Txc);
    Ty.to_cplx(Tyc);
    D2x.to_cplx(D2xc);
    D2y.to_cplx(D2yc);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Half step -- Exact solution
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

      matmul(Lhat,Txc,Nhat);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_vv[1]; j++){
          for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
            complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
            Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_split_h*lambdav*dd1x_r(k))*ncvv;
          }
        }
      }

      matmul_transb(Nhat,Txc,Lhat);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      // Full step --
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

      matmul(Lhat,Tyc,Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.V,v.begin(),Lv);
        ptw_mult_row(lr_sol.V,w.begin(),Lw);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());

        matmul_transb(Lvhat,D2xc,Lhat);
        matmul_transb(Lwhat,D2yc,tmpVhat);

        Lhat +=  tmpVhat;

        matmul(Lhat,Tyc,tmpVhat);

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);
              Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_ee*lambdaw*dd1y_r(k));
              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee*phi1_im(tau_ee*lambdaw*dd1y_r(k))*tmpVhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);
              Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_ee*lambdaw*dd1y_r(k));
              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee*phi1_im(tau_ee*lambdaw*dd1y_r(k))*tmpVhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #endif


        matmul_transb(Nhat,Tyc,Lhat);

        Lhat *= ncvv;
        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

        // Second stage

        ptw_mult_row(lr_sol.V,v.begin(),Lv);
        ptw_mult_row(lr_sol.V,w.begin(),Lw);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());

        matmul_transb(Lvhat,D2xc,Lhat);
        matmul_transb(Lwhat,D2yc,Lvhat);

        Lvhat += Lhat;
        matmul(Lvhat,Tyc,Lhat);

        Lhat -= tmpVhat;

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);

              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee*phi2_im(tau_ee*lambdaw*dd1y_r(k))*Lhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_vv[1]; j++){
            if(j < (N_vv[1]/2)) { mult = j; } else if(j == (N_vv[1]/2)) { mult = 0.0; } else { mult = (j-N_vv[1]); }
            for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
              complex<double> lambdaw = complex<double>(0.0,2.0*M_PI/(lim_vv[3]-lim_vv[2])*mult);

              Nhat(i+j*(N_vv[0]/2+1),k) -= tau_ee*phi2_im(tau_ee*lambdaw*dd1y_r(k))*Lhat(i+j*(N_vv[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Nhat,Tyc,Lhat);

        if(jj != (nsteps_ee - 1)){
          Lhat *= ncvv;
          fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());
        }
      }

      // Half step -- exact solution

      matmul(Lhat,Txc,Nhat);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_vv[1]; j++){
          for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
            complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(lim_vv[1]-lim_vv[0])*i);
            Nhat(i+j*(N_vv[0]/2+1),k) *= exp(tau_split_h*lambdav*dd1x_r(k))*ncvv;
          }
        }
      }

      matmul_transb(Nhat,Txc,Lhat);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_vv);
    transpose_inplace(lr_sol.S);

    // Half step S (until tau/2)

    coeff(lr_sol.V, lr_sol.V, we_v, C1v);
    coeff(lr_sol.V, lr_sol.V, we_w, C1w);

    fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)tmpVhat.begin());

    ptw_mult_row(tmpVhat,lambdav_n.begin(),dVhat_v);
    ptw_mult_row(tmpVhat,lambdaw_n.begin(),dVhat_w);

    fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_v.begin(),dV_v.begin());
    fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)dVhat_w.begin(),dV_w.begin());

    coeff(lr_sol.V, dV_v, h_vv[0]*h_vv[1], C2v);
    coeff(lr_sol.V, dV_w, h_vv[0]*h_vv[1], C2w);

    // Runge Kutta 4
    for(Index jj = 0; jj< nsteps_rk4; jj++){
      matmul_transb(lr_sol.S,C1v,tmpS);
      matmul(D2x,tmpS,tmpS1);
      matmul_transb(lr_sol.S,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS1 += Tw;
      matmul_transb(lr_sol.S,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS1 -= Tw;
      matmul_transb(lr_sol.S,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS1 -= Tw;

      Tv = tmpS1;
      Tv *= (tau_rk4_h/2);
      Tv += lr_sol.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS2);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS2 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS2 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS2 -= Tw;

      Tv = tmpS2;
      Tv *= (tau_rk4_h/2);
      Tv += lr_sol.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS3);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS3 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS3 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS3 -= Tw;

      Tv = tmpS3;
      Tv *= tau_rk4_h;
      Tv += lr_sol.S;

      matmul_transb(Tv,C1v,tmpS);
      matmul(D2x,tmpS,tmpS4);
      matmul_transb(Tv,C1w,tmpS);
      matmul(D2y,tmpS,Tw);
      tmpS4 += Tw;
      matmul_transb(Tv,C2v,tmpS);
      matmul(D1x,tmpS,Tw);
      tmpS4 -= Tw;
      matmul_transb(Tv,C2w,tmpS);
      matmul(D1y,tmpS,Tw);
      tmpS4 -= Tw;

      tmpS2 *= 2.0;
      tmpS3 *= 2.0;

      tmpS1 += tmpS2;
      tmpS1 += tmpS3;
      tmpS1 += tmpS4;
      tmpS1 *= (tau_rk4_h/6.0);

      lr_sol.S += tmpS1;

    }


    // Half step K (until tau/2)

    tmpX = lr_sol.X;
    matmul(tmpX,lr_sol.S,lr_sol.X);

    schur(C1v, Tv, dcv_r, lwork);
    schur(C1w, Tw, dcw_r, lwork);

    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    C2v.to_cplx(C2vc);
    C2w.to_cplx(C2wc);


    for(Index ii = 0; ii < nsteps_split; ii++){
      // Half step -- Exact solution

      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

      matmul(Khat,Tvc,Mhat);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
            complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
            Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_split_h/2.0*lambdax*dcv_r(k))*ncxx;
          }
        }
      }

      matmul_transb(Mhat,Tvc,Khat);

      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

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

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_ee_h*lambday*dcw_r(k));
              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi1_im(-tau_ee_h*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_ee_h*lambday*dcw_r(k));
              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi1_im(-tau_ee_h*lambday*dcw_r(k))*tmpXhat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Mhat,Twc,Khat);

        Khat *= ncxx;
        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

        // Second stage

        ptw_mult_row(lr_sol.X,efx.begin(),Kex);
        ptw_mult_row(lr_sol.X,efy.begin(),Key);

        fftw_execute_dft_r2c(plans_xx[0],Kex.begin(),(fftw_complex*)Kexhat.begin());
        fftw_execute_dft_r2c(plans_xx[0],Key.begin(),(fftw_complex*)Keyhat.begin());

        matmul_transb(Kexhat,C2vc,Khat);
        matmul_transb(Keyhat,C2wc,Kexhat);

        Kexhat += Khat;
        matmul(Kexhat,Twc,Khat);

        Khat -= tmpXhat;

        #ifdef __OPENMP__
        #pragma omp parallel for collapse(2)
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi2_im(-tau_ee_h*lambday*dcw_r(k))*Khat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #else
        for(int k = 0; k < r; k++){
          for(Index j = 0; j < N_xx[1]; j++){
            if(j < (N_xx[1]/2)) { mult = j; } else if(j == (N_xx[1]/2)) { mult = 0.0; } else { mult = (j-N_xx[1]); }
            for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
              complex<double> lambday = complex<double>(0.0,2.0*M_PI/(lim_xx[3]-lim_xx[2])*mult);

              Mhat(i+j*(N_xx[0]/2+1),k) += tau_ee_h*phi2_im(-tau_ee_h*lambday*dcw_r(k))*Khat(i+j*(N_xx[0]/2+1),k);
            }
          }
        }
        #endif

        matmul_transb(Mhat,Twc,Khat);

        if(jj != (nsteps_ee -1)){
          Khat *= ncxx;
          fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin()); // Could be avoided if we do more just one step of the integrator
        }
      }

      // Half step -- exact solution

      matmul(Khat,Tvc,Mhat);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < N_xx[1]; j++){
          for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
            complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(lim_xx[1]-lim_xx[0])*i);
            Mhat(i+j*(N_xx[0]/2+1),k) *= exp(-tau_split_h/2.0*lambdax*dcv_r(k))*ncxx;
          }
        }
      }

      matmul_transb(Mhat,Tvc,Khat);

      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());


    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_xx);

    gt::stop("Restarted integration CPU");

    // Error Mass
    integrate(lr_sol.X,h_xx[0]*h_xx[1],int_x);

    integrate(lr_sol.V,h_vv[0]*h_vv[1],int_v);

    matvec(lr_sol.S,int_v,rho);

    mass = 0.0;
    for(int ii = 0; ii < r; ii++){
      mass += (int_x(ii)*rho(ii));
    }

    err_mass = abs(mass0-mass);

    err_massf << err_mass << endl;

    // Error energy

    integrate(lr_sol.V,we_v2,int_v);

    integrate(lr_sol.V,we_w2,int_v2);

    int_v += int_v2;

    matvec(lr_sol.S,int_v,rho);

    energy = el_energy;
    for(int ii = 0; ii < r; ii++){
      energy += 0.5*(int_x(ii)*rho(ii));
    }

    err_energy = abs(energy0-energy);

    err_energyf << err_energy << endl;
    #endif
    #ifdef __CUDACC__

    // Lie splitting to obtain the electric field

    d_lr_sol_e.X = d_lr_sol.X;
    d_lr_sol_e.S = d_lr_sol.S;
    d_lr_sol_e.V = d_lr_sol.V;

    // Full step K until tau/2

    d_tmpX = d_lr_sol_e.X;

    matmul(d_tmpX,d_lr_sol_e.S,d_lr_sol_e.X);

    gt::start("Lie splitting for electric field GPU");

    integrate(d_lr_sol_e.V,-h_vv[0]*h_vv[1],d_rho);

    matvec(d_lr_sol_e.X,d_rho,d_ef);

    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier_2d<<<(dxxh_mult+n_threads-1)/n_threads,n_threads>>>(dxxh_mult, N_xx[0]/2+1, N_xx[1], d_efhat.begin(), d_lim_xx, ncxx, d_efhatx.begin(),d_efhaty.begin());

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatx.begin(),d_efx.begin());
    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhaty.begin(),d_efy.begin());

    // Electric energy

    cublasDdot (handle_dot, d_efx.num_elements(), d_efx.begin(), 1, d_efx.begin(), 1, d_el_energy_x);
    cublasDdot (handle_dot, d_efy.num_elements(), d_efy.begin(), 1, d_efy.begin(), 1, d_el_energy_y);
    cudaDeviceSynchronize(); // as handle_dot is asynchronous
    ptw_sum<<<1,1>>>(1,d_el_energy_x,d_el_energy_y); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
    scale_unique<<<1,1>>>(d_el_energy_x,0.5*h_xx[0]*h_xx[1]); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_el_energy_CPU,d_el_energy_x,sizeof(double),cudaMemcpyDeviceToHost);

    el_energyGPUf << d_el_energy_CPU << endl;

    // Main of K step

    coeff(d_lr_sol_e.V, d_lr_sol_e.V, d_we_v, d_C1v);
    coeff(d_lr_sol_e.V, d_lr_sol_e.V, d_we_w, d_C1w);

    cufftExecD2Z(d_plans_vv[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_tmpVhat.begin());

    ptw_mult_row_cplx_fourier_2d<<<(dvvh_mult*r+n_threads-1)/n_threads,n_threads>>>(dvvh_mult*r, N_vv[0]/2+1, N_vv[1], d_tmpVhat.begin(), d_lim_vv, ncvv, d_dVhat_v.begin(), d_dVhat_w.begin()); // Very similar, maybe can be put together

    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_v.begin(),d_dV_v.begin());
    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_w.begin(),d_dV_w.begin());

    coeff(d_lr_sol_e.V, d_dV_v, h_vv[0]*h_vv[1], d_C2v);
    coeff(d_lr_sol_e.V, d_dV_w, h_vv[0]*h_vv[1], d_C2w);

    C1v_gpu = d_C1v;
    schur(C1v_gpu, Tv_gpu, dcv_r_gpu, lwork);
    d_Tv = Tv_gpu;
    d_dcv_r = dcv_r_gpu;

    C1w_gpu = d_C1w;
    schur(C1w_gpu, Tw_gpu, dcw_r_gpu, lwork);
    d_Tw = Tw_gpu;
    d_dcw_r = dcw_r_gpu;

    cplx_conv<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(), d_Tv.begin(), d_Tvc.begin());
    cplx_conv<<<(d_Tw.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tw.num_elements(), d_Tw.begin(), d_Twc.begin());
    cplx_conv<<<(d_C2v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2v.num_elements(), d_C2v.begin(), d_C2vc.begin());
    cplx_conv<<<(d_C2w.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2w.num_elements(), d_C2w.begin(), d_C2wc.begin());

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution

      cufftExecD2Z(d_plans_xx[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

      matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h, d_lim_xx, ncxx);

      matmul_transb(d_Mhat,d_Tvc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol_e.X.begin());

      // Full step --
      //#ifdef __FFTW__
      //ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);
      //#else
      cufftExecD2Z(d_plans_xx[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_Khat.begin()); // check if CUDA preserves input and eventually adapt. YES
      //#endif

      matmul(d_Khat,d_Twc,d_Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efy.begin(),d_Key.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());

        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());

        matmul_transb(d_Kexhat,d_C2vc,d_Khat);

        matmul_transb(d_Keyhat,d_C2wc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        matmul(d_Khat,d_Twc,d_tmpXhat);

        exp_euler_fourier_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(),d_dcw_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin());

        matmul_transb(d_Mhat,d_Twc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol_e.X.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_efy.begin(),d_Key.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());

        matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        matmul_transb(d_Keyhat,d_C2wc,d_Kexhat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        matmul(d_Kexhat,d_Twc,d_Khat);

        second_ord_stage_fourier_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(),d_dcw_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin(), d_Khat.begin()); // Very similar, maybe can be put together

        matmul_transb(d_Mhat,d_Twc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol_e.X.begin());

      }
    }

    gram_schmidt_gpu(d_lr_sol_e.X, d_lr_sol_e.S, h_xx[0]*h_xx[1], gen);

    // Full step S until tau/2

    ptw_mult_scal<<<(d_efx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efx.num_elements(), d_efx.begin(), h_xx[0] * h_xx[1], d_we_x.begin());
    ptw_mult_scal<<<(d_efy.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efy.num_elements(), d_efy.begin(), h_xx[0] * h_xx[1], d_we_y.begin());

    coeff(d_lr_sol_e.X, d_lr_sol_e.X, d_we_x, d_D1x);

    coeff(d_lr_sol_e.X, d_lr_sol_e.X, d_we_y, d_D1y);

    cufftExecD2Z(d_plans_xx[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_tmpXhat.begin()); // check if CUDA preserves input and eventually adapt

    ptw_mult_row_cplx_fourier_2d<<<(dxxh_mult*r+n_threads-1)/n_threads,n_threads>>>(dxxh_mult*r, N_xx[0]/2+1, N_xx[1], d_tmpXhat.begin(), d_lim_xx, ncxx, d_dXhat_x.begin(), d_dXhat_y.begin()); // Very similar, maybe can be put together

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_x.begin(),d_dX_x.begin());

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_y.begin(),d_dX_y.begin());

    coeff(d_lr_sol_e.X, d_dX_x, h_xx[0]*h_xx[1], d_D2x);

    coeff(d_lr_sol_e.X, d_dX_y, h_xx[0]*h_xx[1], d_D2y);

    // RK4
    for(Index jj = 0; jj< nsteps_rk4; jj++){
      matmul_transb(d_lr_sol_e.S,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS1);
      matmul_transb(d_lr_sol_e.S,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      matmul_transb(d_lr_sol_e.S,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      matmul_transb(d_lr_sol_e.S,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol_e.S.begin(), d_tmpS1.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS2);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol_e.S.begin(), d_tmpS2.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS3);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(), d_Tv.begin(), tau_rk4_h, d_lr_sol_e.S.begin(), d_tmpS3.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS4);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4_finalcomb<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(), d_lr_sol_e.S.begin(), tau_rk4_h, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_Tw.begin());

    }

    // Full step L until tau/2

    d_tmpV = d_lr_sol_e.V;

    matmul_transb(d_tmpV,d_lr_sol_e.S,d_lr_sol_e.V);

    D1x_gpu = d_D1x;
    schur(D1x_gpu, Tx_gpu, dd1x_r_gpu, lwork);
    d_Tx = Tx_gpu;
    d_dd1x_r = dd1x_r_gpu;

    D1y_gpu = d_D1y;
    schur(D1y_gpu, Ty_gpu, dd1y_r_gpu, lwork);
    d_Ty = Ty_gpu;
    d_dd1y_r = dd1y_r_gpu;

    cplx_conv<<<(d_Tx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tx.num_elements(), d_Tx.begin(), d_Txc.begin());
    cplx_conv<<<(d_Ty.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Ty.num_elements(), d_Ty.begin(), d_Tyc.begin());
    cplx_conv<<<(d_D2x.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2x.num_elements(), d_D2x.begin(), d_D2xc.begin());
    cplx_conv<<<(d_D2y.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2y.num_elements(), d_D2y.begin(), d_D2yc.begin());

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution
      cufftExecD2Z(d_plans_vv[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

      matmul(d_Lhat,d_Txc,d_Nhat);

      exact_sol_exp_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(), d_dd1x_r.begin(), -tau_split_h, d_lim_vv, ncvv);

      matmul_transb(d_Nhat,d_Txc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol_e.V.begin());

      // Full step --
      //#ifdef __FFTW__
      //ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), 1.0/ncvv);
      //#else
      cufftExecD2Z(d_plans_vv[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_Lhat.begin()); // check if CUDA preserves input and eventually adapt
      //#endif

      matmul(d_Lhat,d_Tyc,d_Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_w.begin(),d_Lw.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin()); // check if CUDA preserves input and eventually adapt
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin()); // check if CUDA preserves input and eventually adapt

        matmul_transb(d_Lvhat,d_D2xc,d_Lhat);
        matmul_transb(d_Lwhat,d_D2yc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        matmul(d_Lhat,d_Tyc,d_tmpVhat);

        exp_euler_fourier_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(),d_dd1y_r.begin(),-tau_ee_h, d_lim_vv, d_tmpVhat.begin());

        matmul_transb(d_Nhat,d_Tyc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol_e.V.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_w.begin(),d_Lw.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin()); // check if CUDA preserves input and eventually adapt
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin()); // check if CUDA preserves input and eventually adapt

        matmul_transb(d_Lvhat,d_D2xc,d_Lhat);
        matmul_transb(d_Lwhat,d_D2yc,d_Lvhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lvhat.begin(), d_Lhat.begin());

        matmul(d_Lvhat,d_Tyc,d_Lhat);

        second_ord_stage_fourier_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(),d_dd1y_r.begin(), -tau_ee_h, d_lim_vv, d_tmpVhat.begin(), d_Lhat.begin());

        matmul_transb(d_Nhat,d_Tyc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol_e.V.begin());


      }
    }

    cudaDeviceSynchronize();
    gt::stop("Lie splitting for electric field GPU");

    gt::start("Restarted integration GPU");

    // Electric field at time tau/2

    integrate(d_lr_sol_e.V,-h_vv[0]*h_vv[1],d_rho);

    matvec(d_lr_sol_e.X,d_rho,d_ef);

    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier_2d<<<(dxxh_mult+n_threads-1)/n_threads,n_threads>>>(dxxh_mult, N_xx[0]/2+1, N_xx[1], d_efhat.begin(), d_lim_xx, ncxx, d_efhatx.begin(),d_efhaty.begin());

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatx.begin(),d_efx.begin());
    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhaty.begin(),d_efy.begin());

    // Here I have the electric field at time tau/2, so restart integration

    // Half step K (until tau/2)

    d_tmpX = d_lr_sol.X;
    matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Half step -- Exact solution

      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

      matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h/2.0, d_lim_xx, ncxx);

      matmul_transb(d_Mhat,d_Tvc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      // Full step --
      //#ifdef __FFTW__
      //ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);
      //#else
      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin()); // check if CUDA preserves input and eventually adapt. YES
      //#endif

      matmul(d_Khat,d_Twc,d_Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());

        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());

        matmul_transb(d_Kexhat,d_C2vc,d_Khat);

        matmul_transb(d_Keyhat,d_C2wc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        matmul(d_Khat,d_Twc,d_tmpXhat);

        exp_euler_fourier_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(),d_dcw_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin());

        matmul_transb(d_Mhat,d_Twc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());

        matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        matmul_transb(d_Keyhat,d_C2wc,d_Kexhat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        matmul(d_Kexhat,d_Twc,d_Khat);

        second_ord_stage_fourier_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(),d_dcw_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin(), d_Khat.begin()); // Very similar, maybe can be put together

        matmul_transb(d_Mhat,d_Twc,d_Khat);


        if(jj != (nsteps_ee -1)){
          ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);
          cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());
        }
      }

      // Half step -- Exact solution

      matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h/2.0, d_lim_xx, ncxx);

      matmul_transb(d_Mhat,d_Tvc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());


    }

    gram_schmidt_gpu(d_lr_sol.X, d_lr_sol.S, h_xx[0]*h_xx[1], gen);


    // Half step S (until tau/2)

    ptw_mult_scal<<<(d_efx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efx.num_elements(), d_efx.begin(), h_xx[0] * h_xx[1], d_we_x.begin());
    ptw_mult_scal<<<(d_efy.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efy.num_elements(), d_efy.begin(), h_xx[0] * h_xx[1], d_we_y.begin());

    coeff(d_lr_sol.X, d_lr_sol.X, d_we_x, d_D1x);
    coeff(d_lr_sol.X, d_lr_sol.X, d_we_y, d_D1y);

    cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_tmpXhat.begin()); // check if CUDA preserves input and eventually adapt

    ptw_mult_row_cplx_fourier_2d<<<(dxxh_mult*r+n_threads-1)/n_threads,n_threads>>>(dxxh_mult*r, N_xx[0]/2+1, N_xx[1], d_tmpXhat.begin(), d_lim_xx, ncxx, d_dXhat_x.begin(), d_dXhat_y.begin()); // Very similar, maybe can be put together

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_x.begin(),d_dX_x.begin());
    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_y.begin(),d_dX_y.begin());

    coeff(d_lr_sol.X, d_dX_x, h_xx[0]*h_xx[1], d_D2x);
    coeff(d_lr_sol.X, d_dX_y, h_xx[0]*h_xx[1], d_D2y);

    // RK4
    for(Index jj = 0; jj< nsteps_rk4; jj++){
      matmul_transb(d_lr_sol.S,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS1);
      matmul_transb(d_lr_sol.S,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      matmul_transb(d_lr_sol.S,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      matmul_transb(d_lr_sol.S,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS1.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS2);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS2.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS3);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(), d_Tv.begin(), tau_rk4_h, d_lr_sol.S.begin(), d_tmpS3.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS4);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4_finalcomb<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(), d_lr_sol.S.begin(), tau_rk4_h, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_Tw.begin());

    }

    // Full step L (until tau)

    d_tmpV = d_lr_sol.V;

    matmul_transb(d_tmpV,d_lr_sol.S,d_lr_sol.V);

    D1x_gpu = d_D1x;
    schur(D1x_gpu, Tx_gpu, dd1x_r_gpu, lwork);
    d_Tx = Tx_gpu;
    d_dd1x_r = dd1x_r_gpu;

    D1y_gpu = d_D1y;
    schur(D1y_gpu, Ty_gpu, dd1y_r_gpu, lwork);
    d_Ty = Ty_gpu;
    d_dd1y_r = dd1y_r_gpu;

    cplx_conv<<<(d_Tx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tx.num_elements(), d_Tx.begin(), d_Txc.begin());
    cplx_conv<<<(d_Ty.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Ty.num_elements(), d_Ty.begin(), d_Tyc.begin());
    cplx_conv<<<(d_D2x.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2x.num_elements(), d_D2x.begin(), d_D2xc.begin());
    cplx_conv<<<(d_D2y.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2y.num_elements(), d_D2y.begin(), d_D2yc.begin());

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Half step -- Exact solution
      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

      matmul(d_Lhat,d_Txc,d_Nhat);

      exact_sol_exp_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(), d_dd1x_r.begin(), -tau_split_h, d_lim_vv, ncvv);

      matmul_transb(d_Nhat,d_Txc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      // Full step --
      //#ifdef __FFTW__
      //ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), 1.0/ncvv);
      //#else
      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin()); // check if CUDA preserves input and eventually adapt
      //#endif

      matmul(d_Lhat,d_Tyc,d_Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin()); // check if CUDA preserves input and eventually adapt
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin()); // check if CUDA preserves input and eventually adapt

        matmul_transb(d_Lvhat,d_D2xc,d_Lhat);
        matmul_transb(d_Lwhat,d_D2yc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        matmul(d_Lhat,d_Tyc,d_tmpVhat);

        exp_euler_fourier_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(),d_dd1y_r.begin(),-tau_ee, d_lim_vv, d_tmpVhat.begin());

        matmul_transb(d_Nhat,d_Tyc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin()); // check if CUDA preserves input and eventually adapt
        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin()); // check if CUDA preserves input and eventually adapt

        matmul_transb(d_Lvhat,d_D2xc,d_Lhat);
        matmul_transb(d_Lwhat,d_D2yc,d_Lvhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lvhat.begin(), d_Lhat.begin());

        matmul(d_Lvhat,d_Tyc,d_Lhat);

        second_ord_stage_fourier_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(),d_dd1y_r.begin(), -tau_ee, d_lim_vv, d_tmpVhat.begin(), d_Lhat.begin());

        matmul_transb(d_Nhat,d_Tyc,d_Lhat);

        if(jj != (nsteps_ee - 1)){
          ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);
          cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());
        }
      }

      // Half step -- Exact solution
      matmul(d_Lhat,d_Txc,d_Nhat);

      exact_sol_exp_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(), d_dd1x_r.begin(), -tau_split_h, d_lim_vv, ncvv);

      matmul_transb(d_Nhat,d_Txc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

    }

    gram_schmidt_gpu(d_lr_sol.V, d_lr_sol.S, h_vv[0]*h_vv[1], gen);
    transpose_inplace<<<d_lr_sol.S.num_elements(),1>>>(r,d_lr_sol.S.begin());

    // Half step S (until tau/2)

    coeff(d_lr_sol.V, d_lr_sol.V, d_we_v, d_C1v);
    coeff(d_lr_sol.V, d_lr_sol.V, d_we_w, d_C1w);

    cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_tmpVhat.begin());

    ptw_mult_row_cplx_fourier_2d<<<(dvvh_mult*r+n_threads-1)/n_threads,n_threads>>>(dvvh_mult*r, N_vv[0]/2+1, N_vv[1], d_tmpVhat.begin(), d_lim_vv, ncvv, d_dVhat_v.begin(), d_dVhat_w.begin()); // Very similar, maybe can be put together

    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_v.begin(),d_dV_v.begin());
    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_w.begin(),d_dV_w.begin());

    coeff(d_lr_sol.V, d_dV_v, h_vv[0]*h_vv[1], d_C2v);
    coeff(d_lr_sol.V, d_dV_w, h_vv[0]*h_vv[1], d_C2w);

    // RK4
    for(Index jj = 0; jj< nsteps_rk4; jj++){
      matmul_transb(d_lr_sol.S,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS1);
      matmul_transb(d_lr_sol.S,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      matmul_transb(d_lr_sol.S,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(),d_tmpS1.begin(),d_Tw.begin());
      matmul_transb(d_lr_sol.S,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS1.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS1.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS2);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(),d_tmpS2.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS2.num_elements(), d_Tv.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS2.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS3);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(),d_tmpS3.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4<<<(d_tmpS3.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS3.num_elements(), d_Tv.begin(), tau_rk4_h, d_lr_sol.S.begin(), d_tmpS3.begin(),d_Tw.begin());

      matmul_transb(d_Tv,d_C1v,d_tmpS);
      matmul(d_D2x,d_tmpS,d_tmpS4);
      matmul_transb(d_Tv,d_C1w,d_tmpS);
      matmul(d_D2y,d_tmpS,d_Tw);
      ptw_sum<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2v,d_tmpS);
      matmul(d_D1x,d_tmpS,d_Tw);
      ptw_diff<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(),d_tmpS4.begin(),d_Tw.begin());
      matmul_transb(d_Tv,d_C2w,d_tmpS);
      matmul(d_D1y,d_tmpS,d_Tw);

      rk4_finalcomb<<<(d_tmpS4.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_tmpS4.num_elements(), d_lr_sol.S.begin(), tau_rk4_h, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_Tw.begin());

    }

    // Half step K (until tau/2)

    d_tmpX = d_lr_sol.X;
    matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    C1v_gpu = d_C1v;
    schur(C1v_gpu, Tv_gpu, dcv_r_gpu, lwork);
    d_Tv = Tv_gpu;
    d_dcv_r = dcv_r_gpu;

    C1w_gpu = d_C1w;
    schur(C1w_gpu, Tw_gpu, dcw_r_gpu, lwork);
    d_Tw = Tw_gpu;
    d_dcw_r = dcw_r_gpu;

    cplx_conv<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(), d_Tv.begin(), d_Tvc.begin());
    cplx_conv<<<(d_Tw.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tw.num_elements(), d_Tw.begin(), d_Twc.begin());
    cplx_conv<<<(d_C2v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2v.num_elements(), d_C2v.begin(), d_C2vc.begin());
    cplx_conv<<<(d_C2w.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2w.num_elements(), d_C2w.begin(), d_C2wc.begin());


    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Half step -- Exact solution

      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

      matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h/2.0, d_lim_xx, ncxx);

      matmul_transb(d_Mhat,d_Tvc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      // Full step --
      //#ifdef __FFTW__
      //ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);
      //#else
      cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin()); // check if CUDA preserves input and eventually adapt. YES
      //#endif

      matmul(d_Khat,d_Twc,d_Mhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());

        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());

        matmul_transb(d_Kexhat,d_C2vc,d_Khat);

        matmul_transb(d_Keyhat,d_C2wc,d_tmpXhat);

        ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

        matmul(d_Khat,d_Twc,d_tmpXhat);

        exp_euler_fourier_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(),d_dcw_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin());

        matmul_transb(d_Mhat,d_Twc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

        // Second stage

        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
        ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());

        cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
        cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());

        matmul_transb(d_Kexhat,d_C2vc,d_Khat);
        matmul_transb(d_Keyhat,d_C2wc,d_Kexhat);

        ptw_sum_complex<<<(d_Kexhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Kexhat.num_elements(), d_Kexhat.begin(), d_Khat.begin());

        matmul(d_Kexhat,d_Twc,d_Khat);

        second_ord_stage_fourier_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(),d_dcw_r.begin(),tau_ee_h, d_lim_xx, d_tmpXhat.begin(), d_Khat.begin()); // Very similar, maybe can be put together

        matmul_transb(d_Mhat,d_Twc,d_Khat);


        if(jj != (nsteps_ee -1)){
          ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);
          cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());
        }
      }

      // Half step -- Exact solution

      matmul(d_Khat,d_Tvc,d_Mhat);

      exact_sol_exp_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(), d_dcv_r.begin(), tau_split_h/2.0, d_lim_xx, ncxx);

      matmul_transb(d_Mhat,d_Tvc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());


    }

    gram_schmidt_gpu(d_lr_sol.X, d_lr_sol.S, h_xx[0]*h_xx[1], gen);

    cudaDeviceSynchronize();
    gt::stop("Restarted integration GPU");

    // Error in mass

    integrate(d_lr_sol.X,h_xx[0]*h_xx[1],d_int_x);

    integrate(d_lr_sol.V,h_vv[0]*h_vv[1],d_int_v);

    matvec(d_lr_sol.S,d_int_v,d_rho);

    cublasDdot (handle_dot, r, d_int_x.begin(), 1, d_rho.begin(), 1,d_mass);
    cudaDeviceSynchronize();

    cudaMemcpy(&d_mass_CPU,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    err_mass_CPU = abs(mass0-d_mass_CPU);
    err_massGPUf << err_mass_CPU << endl;

    // Error in energy
    integrate(d_lr_sol.V,d_we_v2,d_int_v);

    integrate(d_lr_sol.V,d_we_w2,d_int_v2);

    ptw_sum<<<(d_int_v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_int_v.num_elements(),d_int_v.begin(),d_int_v2.begin());


    matvec(d_lr_sol.S,d_int_v,d_rho);


    cublasDdot (handle_dot, r, d_int_x.begin(), 1, d_rho.begin(), 1, d_energy);
    cudaDeviceSynchronize();
     scale_unique<<<1,1>>>(d_energy,0.5); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_energy_CPU,d_energy,sizeof(double),cudaMemcpyDeviceToHost);

    err_energy_CPU = abs(energy0-(d_energy_CPU+d_el_energy_CPU));
    err_energyGPUf << err_energy_CPU << endl;

    #endif
    #ifdef __CPU__
    cout << "Electric energy: " << el_energy << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Electric energy GPU: " << d_el_energy_CPU << endl;
    #endif
    #ifdef __CPU__
    cout << "Error in mass: " << err_mass << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Error in mass GPU: " << err_mass_CPU << endl;
    #endif
    #ifdef __CPU__
    cout << "Error in energy: " << err_energy << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Error in energy GPU: " << err_energy_CPU << endl;
    #endif

  }
  #ifdef __CPU__
  el_energyf.close();
  err_massf.close();
  err_energyf.close();
  #endif
  #ifdef __CUDACC__
  cublasDestroy(handle);
  cublasDestroy(handle_dot);

  destroy_plans(plans_d_e);
  destroy_plans(d_plans_xx);
  destroy_plans(d_plans_vv);

  curandDestroyGenerator(gen);

  el_energyGPUf.close();
  err_massGPUf.close();
  err_energyGPUf.close();
  #endif

  //lr_sol = d_lr_sol;
  #ifdef __CPU__
  return lr_sol;
  #else
  lr_sol.X = d_lr_sol.X;
  lr_sol.S = d_lr_sol.S;
  lr_sol.V = d_lr_sol.V;
  cudaDeviceSynchronize();
  return lr_sol;
  #endif
}

int main(){

  #ifdef __OPENMP__
  omp_set_num_threads(n_threads_omp);

  #pragma omp parallel
  {
    if(omp_get_thread_num()==0){
      cout << "Number of threads: " << omp_get_num_threads() << endl;
    }
  }
  #endif

  array<Index,2> N_xx = {32,32}; // Sizes in space
  array<Index,2> N_vv = {32,32}; // Sizes in velocity

  int r = 10; // rank desired

  double tstar = 1.0; // final time

  Index nsteps_ref = 5000;

  vector<Index> nspan = {200,400,600,800,1000};

  int nsteps_split = 1; // Number of time steps internal splitting
  int nsteps_ee = 1; // number of time steps for exponential integrator
  int nsteps_rk4 = 1; // number of time steps for rk4

/*
// Linear Landau
  array<double,4> lim_xx = {0.0,4.0*M_PI,0.0,4.0*M_PI}; // Limits for box [ax,bx] x [ay,by] {ax,bx,ay,by}
  array<double,4> lim_vv = {-6.0,6.0,-6.0,6.0}; // Limits for box [av,bv] x [aw,bw] {av,bv,aw,bw}

  double alpha = 0.01;
  double kappa1 = 0.5;
  double kappa2 = 0.5;
*/
// Two stream instability
  array<double,4> lim_xx = {0.0,10.0*M_PI,0.0,10.0*M_PI}; // Limits for box [ax,bx] x [ay,by] {ax,bx,ay,by}
  array<double,4> lim_vv = {-9.0,9.0,-9.0,9.0}; // Limits for box [av,bv] x [aw,bw] {av,bv,aw,bw}

  double alpha = 0.001;
  double kappa1 = 1.0/5.0;
  double kappa2 = 1.0/5.0;
  double v0 = 2.4;
  double w0 = 2.4;

  // Initial datum generation
  array<double,2> h_xx, h_vv;
  int jj = 0;
  for(int ii = 0; ii < 2; ii++){
    h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
    h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];
    jj+=2;
  }

  vector<const double*> X, V;

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

  multi_array<double,1> vv({dvv_mult});

  for(Index j = 0; j < N_vv[1]; j++){
    for(Index i = 0; i < N_vv[0]; i++){
      double v = lim_vv[0] + i*h_vv[0];
      double w = lim_vv[2] + j*h_vv[1];
      //vv(i+j*N_vv[0]) = (1.0/(2*M_PI)) *exp(-(pow(v,2)+pow(w,2))/2.0);
      vv(i+j*N_vv[0]) = (1.0/(8*M_PI)) *(exp(-pow(v-v0,2)/2.0)+exp(-pow(v+v0,2)/2.0))*(exp(-pow(w-w0,2)/2.0)+exp(-pow(w+w0,2)/2.0));
    }
  }
  V.push_back(vv.begin());

  lr2<double> lr_sol0(r,{dxx_mult,dvv_mult});

  std::function<double(double*,double*)> ip_xx = inner_product_from_const_weight(h_xx[0]*h_xx[1], dxx_mult);
  std::function<double(double*,double*)> ip_vv = inner_product_from_const_weight(h_vv[0]*h_vv[1], dvv_mult);


  // Mandatory to initialize after plan creation if we use FFTW_MEASURE

  multi_array<double,1> ef({dxx_mult});
  multi_array<complex<double>,1> efhat({dxxh_mult});

  array<fftw_plan,2> plans_e = create_plans_2d(N_xx, ef, efhat);

  multi_array<complex<double>,2> Khat({dxxh_mult,r});
  array<fftw_plan,2> plans_xx = create_plans_2d(N_xx, lr_sol0.X, Khat);

  multi_array<complex<double>,2> Lhat({dvvh_mult,r});
  array<fftw_plan,2> plans_vv = create_plans_2d(N_vv, lr_sol0.V, Lhat);

  initialize(lr_sol0, X, V, ip_xx, ip_vv);

  // Computation of reference solution
  lr2<double> lr_sol_fin(r,{dxx_mult,dvv_mult});

  //cout << "First order" << endl;
  //lr_sol_fin = integration_first_order(N_xx,N_vv,r,tstar,nsteps_ref,nsteps_split,nsteps_ee,nsteps_rk4,lim_xx,lim_vv,alpha,kappa1,kappa2,lr_sol0, plans_e, plans_xx, plans_vv);

  //cout << "Second order" << endl;
  lr_sol_fin = integration_second_order(N_xx,N_vv,r,tstar,nsteps_ref,nsteps_split,nsteps_ee,nsteps_rk4,lim_xx,lim_vv,alpha,kappa1,kappa2,lr_sol0, plans_e, plans_xx, plans_vv);

  //cout << gt::sorted_output() << endl;

  multi_array<double,2> refsol({dxx_mult,dvv_mult});
  multi_array<double,2> sol({dxx_mult,dvv_mult});
  multi_array<double,2> tmpsol({dxx_mult,r});

  matmul(lr_sol_fin.X,lr_sol_fin.S,tmpsol);
  matmul_transb(tmpsol,lr_sol_fin.V,refsol);

  ofstream error_order1_2d;
  error_order1_2d.open("../../plots/error_order1_2d.txt");
  error_order1_2d.precision(16);

  ofstream error_order2_2d;
  error_order2_2d.open("../../plots/error_order2_2d.txt");
  error_order2_2d.precision(16);


  error_order1_2d << nspan.size() << endl;
  for(int count = 0; count < nspan.size(); count++){
    error_order1_2d << nspan[count] << endl;
  }

  error_order2_2d << nspan.size() << endl;
  for(int count = 0; count < nspan.size(); count++){
    error_order2_2d << nspan[count] << endl;
  }


  for(int count = 0; count < nspan.size(); count++){

    lr_sol_fin = integration_first_order(N_xx,N_vv,r,tstar,nspan[count],nsteps_split,nsteps_ee,nsteps_rk4,lim_xx,lim_vv,alpha,kappa1,kappa2,lr_sol0, plans_e, plans_xx, plans_vv);

    matmul(lr_sol_fin.X,lr_sol_fin.S,tmpsol);
    matmul_transb(tmpsol,lr_sol_fin.V,sol);

    double error_o1 = 0.0;
    for(int iii = 0; iii < dxx_mult; iii++){
      for(int jjj = 0; jjj < dvv_mult; jjj++){
        double value = abs(refsol(iii,jjj)-sol(iii,jjj));
        if( error_o1 < value){
          error_o1 = value;
        }
      }
    }
    error_order1_2d << error_o1 << endl;

    lr_sol_fin = integration_second_order(N_xx,N_vv,r,tstar,nspan[count],nsteps_split,nsteps_ee,nsteps_rk4,lim_xx,lim_vv,alpha,kappa1,kappa2,lr_sol0, plans_e, plans_xx, plans_vv);

    matmul(lr_sol_fin.X,lr_sol_fin.S,tmpsol);
    matmul_transb(tmpsol,lr_sol_fin.V,sol);

    double error_o2 = 0.0;
    for(int iii = 0; iii < dxx_mult; iii++){
      for(int jjj = 0; jjj < dvv_mult; jjj++){
        double value = abs(refsol(iii,jjj)-sol(iii,jjj));
        if( error_o2 < value){
          error_o2 = value;
        }
      }
    }
    error_order2_2d << error_o2 << endl;

  }

  error_order1_2d.close();
  error_order2_2d.close();

  destroy_plans(plans_e);
  destroy_plans(plans_xx);
  destroy_plans(plans_vv);

  return 0;
}
