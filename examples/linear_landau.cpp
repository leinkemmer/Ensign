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

lr2<double> integration_first_order(Index Nx,Index Nv, int r,double tstar, Index nsteps, int nsteps_ee, int nsteps_rk4, double ax, double bx, double av, double bv, double alpha, double kappa, bool ee_flag, lr2<double> lr_sol, array<fftw_plan,2> plans_e, array<fftw_plan,2> plans_x, array<fftw_plan,2> plans_v){

  double tau = tstar/nsteps;

  double tau_ee = tau / nsteps_ee;
  double tau_rk4 = tau / nsteps_rk4;

  double hx = (bx-ax) / Nx;
  double hv = (bv-av) / Nv;

  multi_array<double,1> v({Nv});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index i = 0; i < Nv; i++){
    v(i) = av + i*hv;
  }

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({Nx});
  multi_array<complex<double>,1> efhat({Nx/2 + 1});

  // For FFT -- Pay attention we have to cast to int as Index seems not to work with fftw_many
  multi_array<complex<double>,2> Khat({Nx/2+1,r});
  multi_array<complex<double>,2> Lhat({Nv/2+1,r});

  std::function<double(double*,double*)> ip_x = inner_product_from_const_weight(hx, Nx);
  std::function<double(double*,double*)> ip_v = inner_product_from_const_weight(hv, Nv);

  multi_array<complex<double>,2> Kehat({Nx/2+1,r});
  multi_array<complex<double>,2> Mhattmp({Nx/2+1,r});

  multi_array<complex<double>,2> Lvhat({Nv/2+1,r});
  multi_array<complex<double>,2> Nhattmp({Nv/2+1,r});

  multi_array<complex<double>,1> lambdax({Nx/2+1});
  multi_array<complex<double>,1> lambdax_n({Nx/2+1});

  double ncx = 1.0 / Nx;

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < (Nx/2 + 1) ; j++){
    lambdax_n(j) = complex<double>(0.0,2.0*M_PI/(bx-ax)*j*ncx);
  }

  multi_array<complex<double>,1> lambdav({Nv/2+1});
  multi_array<complex<double>,1> lambdav_n({Nv/2+1});

  double ncv = 1.0 / Nv;

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < (Nv/2 + 1) ; j++){
    lambdav_n(j) = complex<double>(0.0,2.0*M_PI/(bv-av)*j*ncv);
  }

  // For C coefficients
  multi_array<double,2> C1({r,r});
  multi_array<double,2> C2({r,r});
  multi_array<complex<double>,2> C2c({r,r});

  multi_array<double,1> wv({Nv});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
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

  #ifdef __MKL__
  MKL_INT lwork = -1;
  #else
  int lwork = -1;
  #endif

  schur(T, T, dc_r, lwork);

  // For D coefficients

  multi_array<double,2> D1({r,r});
  multi_array<double,2> D2({r,r});

  multi_array<double,1> wx({Nx});
  multi_array<double,2> dX({Nx,r});

  // Temporary objects
  multi_array<double,2> tmpX({Nx,r});
  multi_array<double,2> tmpV({Nv,r});
  multi_array<double,2> tmpS({r,r});
  multi_array<double,2> tmpS1({r,r});
  multi_array<double,2> tmpS2({r,r});
  multi_array<double,2> tmpS3({r,r});
  multi_array<double,2> tmpS4({r,r});

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

  matmul(lr_sol.X,lr_sol.S,tmpX);
  matvec(tmpX,rho,ef);

  ef += 1.0;

  fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

  efhat(0) = complex<double>(0.0,0.0);

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index ii = 1; ii < (Nx/2+1); ii++){
    complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*ii);

    efhat(ii) /= (lambdax/ncx);
  }

  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhat.begin(),ef.begin());
  energy0 = 0.0;
  #ifdef __OPENMP__
  #pragma omp parallel for reduction(+:energy0)
  #endif
  for(Index ii = 0; ii < Nx; ii++){
    energy0 += 0.5*pow(ef(ii),2)*hx;
  }

  multi_array<double,1> wv2({Nv});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < Nv; j++){
    wv2(j) = pow(v(j),2) * hv;
  }

  coeff_one(lr_sol.V,wv2,int_v);

  matvec(lr_sol.S,int_v,tmp_vec);

  for(int ii = 0; ii < r; ii++){
    energy0 += (0.5*int_x(ii)*tmp_vec(ii));
  }
  #ifdef __CPU__
  ofstream el_energyf;
  ofstream err_massf;
  ofstream err_energyf;

  el_energyf.open("../../plots/el_energy_order1_1d.txt");
  err_massf.open("../../plots/err_mass_order1_1d.txt");
  err_energyf.open("../../plots/err_energy_order1_1d.txt");

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

  lr2<double> d_lr_sol(r,{Nx,Nv}, stloc::device);
  d_lr_sol.X = lr_sol.X;
  d_lr_sol.S = lr_sol.S;
  d_lr_sol.V = lr_sol.V;

  // Electric field
  multi_array<double,1> d_rho({r},stloc::device);
  multi_array<double,1> d_ef({Nx},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhat({Nx/2 + 1},stloc::device);

  array<cufftHandle,2> plans_d_e = create_plans_1d(Nx,1);

  // For FFT
  multi_array<cuDoubleComplex,2> d_Khat({Nx/2+1,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Lhat({Nv/2+1,r},stloc::device);

  array<cufftHandle,2> plans_d_x = create_plans_1d(Nx, r);
  array<cufftHandle,2> plans_d_v = create_plans_1d(Nv, r);

  multi_array<cuDoubleComplex,2> d_Kehat({Nx/2+1,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Mhattmp({Nx/2+1,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Lvhat({Nv/2+1,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Nhattmp({Nv/2+1,r},stloc::device);

  // C coefficients
  multi_array<double,2> d_C1({r,r},stloc::device);
  multi_array<double,2> d_C2({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2c({r,r},stloc::device);

  multi_array<double,1> d_wv({Nv},stloc::device);
  d_wv = wv;

  multi_array<double,2> d_dV({Nv,r},stloc::device);

  // Schur
  multi_array<double,2> C1_gpu({r,r});
  multi_array<double,2> T_gpu({r,r});
  multi_array<double,1> dc_r_gpu({r});

  multi_array<double,1> d_dc_r({r},stloc::device);
  multi_array<double,2> d_T({r,r},stloc::device);
  multi_array<double,2> d_multmp({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Mhat({Nx/2 + 1,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Tc({r,r},stloc::device);

  // D coefficients
  multi_array<double,2> d_D1({r,r},stloc::device);
  multi_array<double,2> d_D2({r,r},stloc::device);

  multi_array<double,1> d_wx({Nx},stloc::device);
  multi_array<double,2> d_dX({Nx,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Nhat({Nv/2 + 1,r},stloc::device);

  multi_array<double,2> D1_gpu({r,r});

  multi_array<double,1> d_v({Nv},stloc::device);
  d_v = v;

  // For plots
  double* d_el_energy;
  cudaMalloc((void**)&d_el_energy,sizeof(double));
  double d_el_energy_CPU;

  multi_array<double,1> d_int_x({r},stloc::device);
  multi_array<double,1> d_int_v({r},stloc::device);
  multi_array<double,1> d_tmp_vec({r},stloc::device);

  double* d_mass;
  cudaMalloc((void**)&d_mass,sizeof(double));
  double d_mass_CPU;
  double err_mass_CPU;

  multi_array<double,1> d_wv2({Nv},stloc::device);
  d_wv2 = wv2;

  double* d_energy;
  cudaMalloc((void**)&d_energy,sizeof(double));
  double d_energy_CPU;
  double err_energy_CPU;

  ofstream el_energyGPUf;
  ofstream err_massGPUf;
  ofstream err_energyGPUf;

  el_energyGPUf.open("../../plots/el_energy_gpu_order1_1d.txt");
  err_massGPUf.open("../../plots/err_mass_gpu_order1_1d.txt");
  err_energyGPUf.open("../../plots/err_energy_gpu_order1_1d.txt");

  el_energyGPUf.precision(16);
  err_massGPUf.precision(16);
  err_energyGPUf.precision(16);

  el_energyGPUf << tstar << endl;
  el_energyGPUf << tau << endl;

  // Temporary
  multi_array<double,2> d_tmpX({Nx,r},stloc::device);
  multi_array<double,2> d_tmpS({r,r},stloc::device);
  multi_array<double,2> d_tmpV({Nv,r},stloc::device);
  multi_array<double,2> d_tmpS1({r,r},stloc::device);
  multi_array<double,2> d_tmpS2({r,r},stloc::device);
  multi_array<double,2> d_tmpS3({r,r},stloc::device);
  multi_array<double,2> d_tmpS4({r,r},stloc::device);

  // For random values generation
  curandGenerator_t gen;

  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,time(0));

  #endif


//  nsteps = 1;
  for(Index i = 0; i < nsteps; i++){


    cout << "Time step " << i + 1 << " on " << nsteps << endl;

    // CPU
    #ifdef __CPU__
    gt::start("Main loop CPU");
    /* K step */
    tmpX = lr_sol.X;

    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field

    coeff_one(lr_sol.V,-hv,rho);

    matvec(lr_sol.X,rho,ef);

    ef += 1.0;

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    efhat(0) = complex<double>(0.0,0.0);
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
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

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

          Mhat(j,k) *= exp(-tau_ee*lambdax*dc_r(k));
          Mhat(j,k) += tau_ee*phi1_im(-tau_ee*lambdax*dc_r(k))*Mhattmp(j,k);
        }
      }

      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

      // Second stage

      ptw_mult_row(lr_sol.X,ef.begin(),lr_sol.X);

      fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Kehat.begin());

      matmul_transb(Kehat,C2c,Khat);

      matmul(Khat,Tc,Kehat);

      Kehat -= Mhattmp;

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

          Mhat(j,k) += tau_ee*phi2_im(-tau_ee*lambdax*dc_r(k))*Kehat(j,k);
        }
      }

      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

    }
    gram_schmidt(lr_sol.X, lr_sol.S, ip_x);

    /* S step */

    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index ii = 0; ii < Nx; ii++){
      wx(ii) = hx*ef(ii);
    }

    coeff(lr_sol.X, lr_sol.X, wx.begin(), D1);

    fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    ptw_mult_row(Khat,lambdax_n.begin(),Khat);

    fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),dX.begin());

    coeff(lr_sol.X, dX, hx, D2);

    if(ee_flag){
      // Explicit Euler
      for(int j = 0; j< nsteps_rk4; j++){
        matmul_transb(lr_sol.S,C1,tmpS);
        matmul(D2,tmpS,T);

        matmul_transb(lr_sol.S,C2,tmpS);
        matmul(D1,tmpS,multmp);

        T -= multmp;
        T *= tau_rk4;
        lr_sol.S += T;
      }
    }else{
      // RK4
      for(int j = 0; j< nsteps_rk4; j++){
        matmul_transb(lr_sol.S,C1,tmpS);
        matmul(D2,tmpS,tmpS1);

        matmul_transb(lr_sol.S,C2,tmpS);
        matmul(D1,tmpS,multmp);

        tmpS1 -= multmp;

        T = tmpS1;
        T *= (tau_rk4/2);
        T += lr_sol.S;

        matmul_transb(T,C1,tmpS);
        matmul(D2,tmpS,tmpS2);

        matmul_transb(T,C2,tmpS);
        matmul(D1,tmpS,multmp);

        tmpS2 -= multmp;

        T = tmpS2;
        T *= (tau_rk4/2);
        T += lr_sol.S;

        matmul_transb(T,C1,tmpS);
        matmul(D2,tmpS,tmpS3);

        matmul_transb(T,C2,tmpS);
        matmul(D1,tmpS,multmp);

        tmpS3 -= multmp;

        T = tmpS3;
        T *= tau_rk4;
        T += lr_sol.S;

        matmul_transb(T,C1,tmpS);
        matmul(D2,tmpS,tmpS4);

        matmul_transb(T,C2,tmpS);
        matmul(D1,tmpS,multmp);

        tmpS4 -= multmp;

        tmpS2 *= 2.0;
        tmpS3 *= 2.0;

        tmpS1 += tmpS2;
        tmpS1 += tmpS3;
        tmpS1 += tmpS4;
        tmpS1 *= (tau_rk4/6.0);

        lr_sol.S += tmpS1;

      }
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

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nv/2 + 1); j++){
          complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(bv-av)*j);

          Nhat(j,k) *= exp(tau_ee*lambdav*dc_r(k));
          Nhat(j,k) += -tau_ee*phi1_im(tau_ee*lambdav*dc_r(k))*Nhattmp(j,k);
        }
      }

      matmul_transb(Nhat,Tc,Lhat);

      Lhat *= ncv;

      fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      // Second stage

      ptw_mult_row(lr_sol.V,v.begin(),lr_sol.V);

      fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lvhat.begin());

      matmul_transb(Lvhat,C2c,Lhat);

      matmul(Lhat,Tc,Lvhat);

      Lvhat -= Nhattmp;

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nv/2 + 1); j++){
          complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(bv-av)*j);

          Nhat(j,k) += -tau_ee*phi2_im(tau_ee*lambdav*dc_r(k))*Lvhat(j,k);
        }
      }

      matmul_transb(Nhat,Tc,Lhat);

      Lhat *= ncv;

      fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());


    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_v);

    transpose_inplace(lr_sol.S);

    gt::stop("Main loop CPU");

    // Electric energy
    el_energy = 0.0;
    #ifdef __OPENMP__
    #pragma omp parallel for reduction(+:el_energy)
    #endif
    for(Index ii = 0; ii < Nx; ii++){
      el_energy += 0.5*pow(ef(ii),2)*hx;
    }

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

    err_massf << err_mass << endl;

    coeff_one(lr_sol.V,wv2,int_v);

    matvec(lr_sol.S,int_v,tmp_vec);

    energy = el_energy;
    for(int ii = 0; ii < r; ii++){
      energy += (0.5*int_x(ii)*tmp_vec(ii));
    }

    err_energy = abs(energy0-energy);

    err_energyf << err_energy << endl;

    #endif
    // GPU
    #ifdef __CUDACC__
    // K step

    gt::start("Main loop GPU");

    d_tmpX = d_lr_sol.X;

    matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    coeff_one(d_lr_sol.V,-hv,d_rho);

    matvec(d_lr_sol.X,d_rho,d_ef);

    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier<<<(Nx/2+1+n_threads-1)/n_threads,n_threads>>>(Nx/2+1, d_efhat.begin(), ax, bx, Nx); //128 threads blocks as needed

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhat.begin(),d_ef.begin());

    coeff(d_lr_sol.V, d_lr_sol.V, d_wv.begin(), d_C1);

    cufftExecD2Z(plans_d_v[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

    ptw_mult_row_cplx_fourier<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.shape()[0], Nv, d_Lhat.begin(), av, bv);

    cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_dV.begin());

    coeff(d_lr_sol.V, d_dV, hv, d_C2);

    C1_gpu = d_C1;
    schur(C1_gpu, T_gpu, dc_r_gpu, lwork);
    d_T = T_gpu;
    d_dc_r = dc_r_gpu;

    cplx_conv<<<(d_T.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_T.num_elements(), d_T.begin(), d_Tc.begin());

    cplx_conv<<<(d_C2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2.num_elements(), d_C2.begin(), d_C2c.begin());

    cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

    matmul(d_Khat,d_Tc,d_Mhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_ef.begin(),d_lr_sol.X.begin());

      cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Kehat.begin());

      matmul_transb(d_Kehat,d_C2c,d_Khat);

      matmul(d_Khat,d_Tc,d_Mhattmp);

      exp_euler_fourier<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), d_Mhat.shape()[0], d_Mhat.begin(), d_dc_r.begin(), tau_ee, d_Mhattmp.begin(), ax, bx);

      matmul_transb(d_Mhat,d_Tc,d_Khat);

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(),d_Khat.begin(),ncx);

      cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      // Second stage

      ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_ef.begin(),d_lr_sol.X.begin());

      cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Kehat.begin());

      matmul_transb(d_Kehat,d_C2c,d_Khat);

      matmul(d_Khat,d_Tc,d_Kehat);

      second_ord_stage_fourier<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), d_Mhat.shape()[0], d_Mhat.begin(), d_dc_r.begin(), tau_ee, d_Mhattmp.begin(), d_Kehat.begin(), ax, bx);

      matmul_transb(d_Mhat,d_Tc,d_Khat);

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(),d_Khat.begin(),ncx);

      cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

    }


    gram_schmidt_gpu(d_lr_sol.X, d_lr_sol.S, hx, gen);


    // S step

    ptw_mult_scal<<<(d_ef.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_ef.num_elements(), d_ef.begin(), hx, d_wx.begin());

    coeff(d_lr_sol.X, d_lr_sol.X, d_wx.begin(), d_D1);

    cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

    ptw_mult_row_cplx_fourier<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.shape()[0], Nx, d_Khat.begin(), ax, bx);

    cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_dX.begin());

    coeff(d_lr_sol.X, d_dX, hx, d_D2);

    if(ee_flag){
      // Explicit Euler
      for(int j = 0; j< nsteps_rk4; j++){
        matmul_transb(d_lr_sol.S,d_C1,d_tmpS);
        matmul(d_D2,d_tmpS,d_T);

        matmul_transb(d_lr_sol.S,d_C2,d_tmpS);
        matmul(d_D1,d_tmpS,d_multmp);

        expl_euler<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_lr_sol.S.begin(), tau_rk4, d_T.begin(), d_multmp.begin());
      }
    }else{
      // RK4
      for(int j = 0; j< nsteps_rk4; j++){
        matmul_transb(d_lr_sol.S,d_C1,d_tmpS);
        matmul(d_D2,d_tmpS,d_tmpS1);

        matmul_transb(d_lr_sol.S,d_C2,d_tmpS);
        matmul(d_D1,d_tmpS,d_multmp);

        rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4/2.0, d_lr_sol.S.begin(), d_tmpS1.begin(),d_multmp.begin());

        matmul_transb(d_T,d_C1,d_tmpS);
        matmul(d_D2,d_tmpS,d_tmpS2);

        matmul_transb(d_T,d_C2,d_tmpS);
        matmul(d_D1,d_tmpS,d_multmp);

        rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4/2.0, d_lr_sol.S.begin(), d_tmpS2.begin(),d_multmp.begin());

        matmul_transb(d_T,d_C1,d_tmpS);
        matmul(d_D2,d_tmpS,d_tmpS3);

        matmul_transb(d_T,d_C2,d_tmpS);
        matmul(d_D1,d_tmpS,d_multmp);

        rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4, d_lr_sol.S.begin(), d_tmpS3.begin(),d_multmp.begin());

        matmul_transb(d_T,d_C1,d_tmpS);
        matmul(d_D2,d_tmpS,d_tmpS4);

        matmul_transb(d_T,d_C2,d_tmpS);
        matmul(d_D1,d_tmpS,d_multmp);

        rk4_finalcomb<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_lr_sol.S.begin(), tau_rk4, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_multmp.begin());
      }
    }


    // L step
    d_tmpV = d_lr_sol.V;

    matmul_transb(d_tmpV,d_lr_sol.S,d_lr_sol.V);

    D1_gpu = d_D1;
    schur(D1_gpu, T_gpu, dc_r_gpu, lwork);
    d_T = T_gpu;
    d_dc_r = dc_r_gpu;

    cplx_conv<<<(d_T.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_T.num_elements(), d_T.begin(), d_Tc.begin());

    cplx_conv<<<(d_D2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2.num_elements(), d_D2.begin(), d_C2c.begin());

    cufftExecD2Z(plans_d_v[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

    matmul(d_Lhat,d_Tc,d_Nhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_lr_sol.V.begin());

      cufftExecD2Z(plans_d_v[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lvhat.begin());

      matmul_transb(d_Lvhat,d_C2c,d_Lhat);

      matmul(d_Lhat,d_Tc,d_Nhattmp);

      exp_euler_fourier<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), d_Nhat.shape()[0], d_Nhat.begin(), d_dc_r.begin(), -tau_ee, d_Nhattmp.begin(), av, bv);

      matmul_transb(d_Nhat,d_Tc,d_Lhat);

      ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(),d_Lhat.begin(),ncv);

      cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      // Second stage

      ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_lr_sol.V.begin());

      cufftExecD2Z(plans_d_v[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lvhat.begin());

      matmul_transb(d_Lvhat,d_C2c,d_Lhat);

      matmul(d_Lhat,d_Tc,d_Lvhat);

      second_ord_stage_fourier<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), d_Nhat.shape()[0], d_Nhat.begin(), d_dc_r.begin(), -tau_ee, d_Nhattmp.begin(), d_Lvhat.begin(), av, bv);

      matmul_transb(d_Nhat,d_Tc,d_Lhat);

      ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(),d_Lhat.begin(),ncv);

      cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

    }

    gram_schmidt_gpu(d_lr_sol.V, d_lr_sol.S, hv, gen);

    transpose_inplace<<<d_lr_sol.S.num_elements(),1>>>(r,d_lr_sol.S.begin());

    cudaDeviceSynchronize();
    gt::stop("Main loop GPU");

    cublasDdot (handle_dot, Nx, d_ef.begin(), 1, d_ef.begin(), 1, d_el_energy);
    cudaDeviceSynchronize();
    scale_unique<<<1,1>>>(d_el_energy,0.5*hx); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_el_energy_CPU,d_el_energy,sizeof(double),cudaMemcpyDeviceToHost);

    el_energyGPUf << d_el_energy_CPU << endl;

    // Error mass
    coeff_one(d_lr_sol.X,hx,d_int_x);

    coeff_one(d_lr_sol.V,hv,d_int_v);

    matvec(d_lr_sol.S,d_int_v,d_tmp_vec);

    cublasDdot (handle_dot, r, d_int_x.begin(), 1, d_tmp_vec.begin(), 1,d_mass);
    cudaDeviceSynchronize();
    cudaMemcpy(&d_mass_CPU,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    err_mass_CPU = abs(mass0-d_mass_CPU);

    err_massGPUf << err_mass_CPU << endl;

    coeff_one(d_lr_sol.V,d_wv2,d_int_v);

    matvec(d_lr_sol.S,d_int_v,d_tmp_vec);

    cublasDdot (handle_dot, r, d_int_x.begin(), 1, d_tmp_vec.begin(), 1,d_energy);
    cudaDeviceSynchronize();
    scale_unique<<<1,1>>>(d_energy,0.5); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_energy_CPU,d_energy,sizeof(double),cudaMemcpyDeviceToHost);

    err_energy_CPU = abs(energy0-(d_energy_CPU+d_el_energy_CPU));

    err_energyGPUf << err_energy_CPU << endl;

    #endif
    #ifdef __CPU__
    cout << "Electric energy CPU: " << el_energy << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Electric energy GPU: " << d_el_energy_CPU << endl;
    #endif
    #ifdef __CPU__
    cout << "Error in mass CPU: " << err_mass << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Error in mass GPU: " << err_mass_CPU << endl;
    #endif
    #ifdef __CPU__
    cout << "Error in energy CPU: " << err_energy << endl;
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
  destroy_plans(plans_d_x);
  destroy_plans(plans_d_v);

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
  return lr_sol;
  #endif
}

lr2<double> integration_second_order(Index Nx,Index Nv, int r,double tstar, Index nsteps, int nsteps_ee, int nsteps_rk4, double ax, double bx, double av, double bv, double alpha, double kappa, lr2<double> lr_sol, array<fftw_plan,2> plans_e, array<fftw_plan,2> plans_x, array<fftw_plan,2> plans_v){

  double tau = tstar/nsteps;
  double tau_h = tau/2.0;

  double tau_ee = tau / nsteps_ee;
  double tau_ee_h = tau_h / nsteps_ee;

  double tau_rk4_h = tau_h / nsteps_rk4;

  double hx = (bx-ax) / Nx;
  double hv = (bv-av) / Nv;

  multi_array<double,1> v({Nv});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index i = 0; i < Nv; i++){
    v(i) = av + i*hv;
  }

  // For Electric field

  multi_array<double,1> rho({r});
  multi_array<double,1> ef({Nx});
  multi_array<complex<double>,1> efhat({Nx/2 + 1});

  multi_array<complex<double>,2> Khat({Nx/2+1,r});
  multi_array<complex<double>,2> Lhat({Nv/2+1,r});

  std::function<double(double*,double*)> ip_x = inner_product_from_const_weight(hx, Nx);
  std::function<double(double*,double*)> ip_v = inner_product_from_const_weight(hv, Nv);

  multi_array<complex<double>,2> Kehat({Nx/2+1,r});
  multi_array<complex<double>,2> Mhattmp({Nx/2+1,r});

  multi_array<complex<double>,2> Lvhat({Nv/2+1,r});
  multi_array<complex<double>,2> Nhattmp({Nv/2+1,r});

  multi_array<complex<double>,1> lambdax({Nx/2+1});
  multi_array<complex<double>,1> lambdax_n({Nx/2+1});

  double ncx = 1.0 / Nx;

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < (Nx/2 + 1) ; j++){
    lambdax_n(j) = complex<double>(0.0,2.0*M_PI/(bx-ax)*j*ncx);
  }

  multi_array<complex<double>,1> lambdav({Nv/2+1});
  multi_array<complex<double>,1> lambdav_n({Nv/2+1});

  double ncv = 1.0 / Nv;

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < (Nv/2 + 1) ; j++){
    lambdav_n(j) = complex<double>(0.0,2.0*M_PI/(bv-av)*j*ncv);
  }

  // For C coefficients
  multi_array<double,2> C1({r,r});
  multi_array<double,2> C2({r,r});
  multi_array<complex<double>,2> C2c({r,r});

  multi_array<double,1> wv({Nv});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < Nv; j++){
    wv(j) = v(j) * hv;
  }

  multi_array<double,2> dV({Nv,r});

  // For Schur decomposition
  multi_array<double,1> dc_r({r});
  multi_array<double,1> dd_r({r});

  multi_array<double,2> T({r,r});
  multi_array<double,2> W({r,r});

  multi_array<double,2> multmp({r,r});

  multi_array<complex<double>,2> Mhat({Nx/2 + 1,r});
  multi_array<complex<double>,2> Nhat({Nv/2 + 1,r});
  multi_array<complex<double>,2> Tc({r,r});
  multi_array<complex<double>,2> Wc({r,r});

  #ifdef __MKL__
  MKL_INT lwork = -1;
  #else
  int lwork = -1;
  #endif

  schur(T, T, dc_r, lwork);

  // For D coefficients

  multi_array<double,2> D1({r,r});
  multi_array<double,2> D2({r,r});
  multi_array<complex<double>,2> D2c({r,r});

  multi_array<double,1> wx({Nx});
  multi_array<double,2> dX({Nx,r});

  // Temporary objects
  multi_array<double,2> tmpX({Nx,r});
  multi_array<double,2> tmpV({Nv,r});
  multi_array<double,2> tmpS({r,r});
  multi_array<double,2> tmpS1({r,r});
  multi_array<double,2> tmpS2({r,r});
  multi_array<double,2> tmpS3({r,r});
  multi_array<double,2> tmpS4({r,r});

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

  matmul(lr_sol.X,lr_sol.S,tmpX);
  matvec(tmpX,rho,ef);

  ef += 1.0;

  fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

  efhat(0) = complex<double>(0.0,0.0);
  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index ii = 1; ii < (Nx/2+1); ii++){
    complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*ii);

    efhat(ii) /= (lambdax/ncx);
  }

  fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhat.begin(),ef.begin());
  energy0 = 0.0;
  #ifdef __OPENMP__
  #pragma omp parallel for reduction(+:energy0)
  #endif
  for(Index ii = 0; ii < Nx; ii++){
    energy0 += 0.5*pow(ef(ii),2)*hx;
  }

  multi_array<double,1> wv2({Nv});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < Nv; j++){
    wv2(j) = pow(v(j),2) * hv;
  }

  coeff_one(lr_sol.V,wv2,int_v);

  matvec(lr_sol.S,int_v,tmp_vec);

  for(int ii = 0; ii < r; ii++){
    energy0 += (0.5*int_x(ii)*tmp_vec(ii));
  }
  #ifdef __CPU__
  ofstream el_energyf;
  ofstream err_massf;
  ofstream err_energyf;

  el_energyf.open("../../plots/el_energy_order2_1d.txt");
  err_massf.open("../../plots/err_mass_order2_1d.txt");
  err_energyf.open("../../plots/err_energy_order2_1d.txt");

  el_energyf.precision(16);
  err_massf.precision(16);
  err_energyf.precision(16);

  el_energyf << tstar << endl;
  el_energyf << tau << endl;
  #endif
  // Additional stuff for second order
  lr2<double> lr_sol_e(r,{Nx,Nv});


  //// FOR GPU ////
  #ifdef __CUDACC__
  cublasCreate (&handle);
  cublasCreate (&handle_dot);
  cublasSetPointerMode(handle_dot, CUBLAS_POINTER_MODE_DEVICE);

  lr2<double> d_lr_sol(r,{Nx,Nv}, stloc::device);
  d_lr_sol.X = lr_sol.X;
  d_lr_sol.S = lr_sol.S;
  d_lr_sol.V = lr_sol.V;

  // Electric field
  multi_array<double,1> d_rho({r},stloc::device);
  multi_array<double,1> d_ef({Nx},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhat({Nx/2 + 1},stloc::device);

  array<cufftHandle,2> plans_d_e = create_plans_1d(Nx,1);

  // For FFT
  multi_array<cuDoubleComplex,2> d_Khat({Nx/2+1,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Lhat({Nv/2+1,r},stloc::device);

  array<cufftHandle,2> plans_d_x = create_plans_1d(Nx, r);
  array<cufftHandle,2> plans_d_v = create_plans_1d(Nv, r);

  multi_array<cuDoubleComplex,2> d_Kehat({Nx/2+1,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Mhattmp({Nx/2+1,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Lvhat({Nv/2+1,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Nhattmp({Nv/2+1,r},stloc::device);

  // C coefficients
  multi_array<double,2> d_C1({r,r},stloc::device);
  multi_array<double,2> d_C2({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2c({r,r},stloc::device);

  multi_array<double,1> d_wv({Nv},stloc::device);
  d_wv = wv;

  multi_array<double,2> d_dV({Nv,r},stloc::device);

  // Schur
  multi_array<double,2> C1_gpu({r,r});
  multi_array<double,2> T_gpu({r,r});
  multi_array<double,2> W_gpu({r,r});
  multi_array<double,1> dc_r_gpu({r});
  multi_array<double,1> dd_r_gpu({r});

  multi_array<double,1> d_dc_r({r},stloc::device);
  multi_array<double,1> d_dd_r({r},stloc::device);
  multi_array<double,2> d_T({r,r},stloc::device);
  multi_array<double,2> d_W({r,r},stloc::device);
  multi_array<double,2> d_multmp({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Mhat({Nx/2 + 1,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Tc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Wc({r,r},stloc::device);

  // D coefficients
  multi_array<double,2> d_D1({r,r},stloc::device);
  multi_array<double,2> d_D2({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_D2c({r,r},stloc::device);

  multi_array<double,1> d_wx({Nx},stloc::device);
  multi_array<double,2> d_dX({Nx,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Nhat({Nv/2 + 1,r},stloc::device);

  multi_array<double,2> D1_gpu({r,r});

  multi_array<double,1> d_v({Nv},stloc::device);
  d_v = v;

  // For plots
  double* d_el_energy;
  cudaMalloc((void**)&d_el_energy,sizeof(double));
  double d_el_energy_CPU;

  multi_array<double,1> d_int_x({r},stloc::device);
  multi_array<double,1> d_int_v({r},stloc::device);
  multi_array<double,1> d_tmp_vec({r},stloc::device);

  double* d_mass;
  cudaMalloc((void**)&d_mass,sizeof(double));
  double d_mass_CPU;
  double err_mass_CPU;

  multi_array<double,1> d_wv2({Nv},stloc::device);
  d_wv2 = wv2;

  double* d_energy;
  cudaMalloc((void**)&d_energy,sizeof(double));
  double d_energy_CPU;
  double err_energy_CPU;

  ofstream el_energyGPUf;
  ofstream err_massGPUf;
  ofstream err_energyGPUf;

  el_energyGPUf.open("../../plots/el_energy_gpu_order2_1d.txt");
  err_massGPUf.open("../../plots/err_mass_gpu_order2_1d.txt");
  err_energyGPUf.open("../../plots/err_energy_gpu_order2_1d.txt");

  el_energyGPUf.precision(16);
  err_massGPUf.precision(16);
  err_energyGPUf.precision(16);

  el_energyGPUf << tstar << endl;
  el_energyGPUf << tau << endl;

  // Temporary
  multi_array<double,2> d_tmpX({Nx,r},stloc::device);
  multi_array<double,2> d_tmpS({r,r},stloc::device);
  multi_array<double,2> d_tmpV({Nv,r},stloc::device);
  multi_array<double,2> d_tmpS1({r,r},stloc::device);
  multi_array<double,2> d_tmpS2({r,r},stloc::device);
  multi_array<double,2> d_tmpS3({r,r},stloc::device);
  multi_array<double,2> d_tmpS4({r,r},stloc::device);

  // Additional stuff for second order
  lr2<double> d_lr_sol_e(r,{Nx,Nv}, stloc::device);

  // For random values generation
  curandGenerator_t gen;

  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,time(0));


  #endif

  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i + 1 << " on " << nsteps << endl;
    #ifdef __CPU__
    /* Lie splitting to obtain the electric field */

    lr_sol_e.X = lr_sol.X;
    lr_sol_e.S = lr_sol.S;
    lr_sol_e.V = lr_sol.V;

    /* Half step K */

    gt::start("Lie splitting for electric field CPU");

    tmpX = lr_sol_e.X;

    matmul(tmpX,lr_sol_e.S,lr_sol_e.X);

    // Electric field

    coeff_one(lr_sol_e.V,-hv,rho);

    matvec(lr_sol_e.X,rho,ef);

    ef += 1.0;

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    efhat(0) = complex<double>(0.0,0.0);
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index ii = 1; ii < (Nx/2+1); ii++){
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*ii);

      efhat(ii) /= (lambdax/ncx);
    }

    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhat.begin(),ef.begin());

    // Electric energy
    el_energy = 0.0;
    #ifdef __OPENMP__
    #pragma omp parallel for reduction(+:el_energy)
    #endif
    for(Index ii = 0; ii < Nx; ii++){
      el_energy += 0.5*pow(ef(ii),2)*hx;
    }

    el_energyf << el_energy << endl;


    // Main of K step

    coeff(lr_sol_e.V, lr_sol_e.V, wv.begin(), C1);

    fftw_execute_dft_r2c(plans_v[0],lr_sol_e.V.begin(),(fftw_complex*)Lhat.begin());

    ptw_mult_row(Lhat,lambdav_n.begin(),Lhat);

    fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),dV.begin());

    coeff(lr_sol_e.V, dV, hv, C2);

    schur(C1, T, dc_r, lwork);

    T.to_cplx(Tc);

    C2.to_cplx(C2c);

    fftw_execute_dft_r2c(plans_x[0],lr_sol_e.X.begin(),(fftw_complex*)Khat.begin());

    matmul(Khat,Tc,Mhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row(lr_sol_e.X,ef.begin(),lr_sol_e.X);

      fftw_execute_dft_r2c(plans_x[0],lr_sol_e.X.begin(),(fftw_complex*)Kehat.begin());

      matmul_transb(Kehat,C2c,Khat);

      matmul(Khat,Tc,Mhattmp);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

          Mhat(j,k) *= exp(-tau_ee_h*lambdax*dc_r(k));
          Mhat(j,k) += tau_ee_h*phi1_im(-tau_ee_h*lambdax*dc_r(k))*Mhattmp(j,k);
        }
      }

      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol_e.X.begin());

      // Second stage

      ptw_mult_row(lr_sol_e.X,ef.begin(),lr_sol_e.X);

      fftw_execute_dft_r2c(plans_x[0],lr_sol_e.X.begin(),(fftw_complex*)Kehat.begin());

      matmul_transb(Kehat,C2c,Khat);

      matmul(Khat,Tc,Kehat);

      Kehat -= Mhattmp;

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

          Mhat(j,k) += tau_ee_h*phi2_im(-tau_ee_h*lambdax*dc_r(k))*Kehat(j,k);
        }
      }

      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol_e.X.begin());

    }

    gram_schmidt(lr_sol_e.X, lr_sol_e.S, ip_x);


    /* Half step S */
    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index ii = 0; ii < Nx; ii++){
      wx(ii) = hx*ef(ii);
    }

    coeff(lr_sol_e.X, lr_sol_e.X, wx.begin(), D1);

    fftw_execute_dft_r2c(plans_x[0],lr_sol_e.X.begin(),(fftw_complex*)Khat.begin());

    ptw_mult_row(Khat,lambdax_n.begin(),Khat);

    fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),dX.begin());

    coeff(lr_sol_e.X, dX, hx, D2);

    // RK4
    for(int j = 0; j< nsteps_rk4; j++){
      matmul_transb(lr_sol_e.S,C1,tmpS);
      matmul(D2,tmpS,tmpS1);

      matmul_transb(lr_sol_e.S,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS1 -= multmp;

      T = tmpS1;
      T *= (tau_rk4_h/2);
      T += lr_sol_e.S;

      matmul_transb(T,C1,tmpS);
      matmul(D2,tmpS,tmpS2);

      matmul_transb(T,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS2 -= multmp;

      T = tmpS2;
      T *= (tau_rk4_h/2);
      T += lr_sol_e.S;

      matmul_transb(T,C1,tmpS);
      matmul(D2,tmpS,tmpS3);

      matmul_transb(T,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS3 -= multmp;

      T = tmpS3;
      T *= tau_rk4_h;
      T += lr_sol_e.S;

      matmul_transb(T,C1,tmpS);
      matmul(D2,tmpS,tmpS4);

      matmul_transb(T,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS4 -= multmp;

      tmpS2 *= 2.0;
      tmpS3 *= 2.0;

      tmpS1 += tmpS2;
      tmpS1 += tmpS3;
      tmpS1 += tmpS4;
      tmpS1 *= (tau_rk4_h/6.0);

      lr_sol_e.S += tmpS1;
    }

    /* Half step L */

    tmpV = lr_sol_e.V;
    matmul_transb(tmpV,lr_sol_e.S,lr_sol_e.V);

    schur(D1, W, dd_r, lwork);

    W.to_cplx(Wc);
    D2.to_cplx(D2c);

    fftw_execute_dft_r2c(plans_v[0],lr_sol_e.V.begin(),(fftw_complex*)Lhat.begin());

    matmul(Lhat,Wc,Nhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row(lr_sol_e.V,v.begin(),lr_sol_e.V);

      fftw_execute_dft_r2c(plans_v[0],lr_sol_e.V.begin(),(fftw_complex*)Lvhat.begin());

      matmul_transb(Lvhat,D2c,Lhat);

      matmul(Lhat,Wc,Nhattmp);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nv/2 + 1); j++){
          complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(bv-av)*j);

          Nhat(j,k) *= exp(tau_ee_h*lambdav*dd_r(k));
          Nhat(j,k) += -tau_ee_h*phi1_im(tau_ee_h*lambdav*dd_r(k))*Nhattmp(j,k);
        }
      }

      matmul_transb(Nhat,Wc,Lhat);

      Lhat *= ncv;

      fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),lr_sol_e.V.begin());

      // Second stage

      ptw_mult_row(lr_sol_e.V,v.begin(),lr_sol_e.V);

      fftw_execute_dft_r2c(plans_v[0],lr_sol_e.V.begin(),(fftw_complex*)Lvhat.begin());

      matmul_transb(Lvhat,D2c,Lhat);

      matmul(Lhat,Wc,Lvhat);

      Lvhat -= Nhattmp;

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nv/2 + 1); j++){
          complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(bv-av)*j);

          Nhat(j,k) += -tau_ee_h*phi2_im(tau_ee_h*lambdav*dd_r(k))*Lvhat(j,k);
        }
      }

      matmul_transb(Nhat,Wc,Lhat);

      Lhat *= ncv;

      fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),lr_sol_e.V.begin());

    }

    gt::stop("Lie splitting for electric field CPU");

    gt::start("Restarted integration CPU");

    // Electric field at tau/2
    coeff_one(lr_sol_e.V,-hv,rho);

    matvec(lr_sol_e.X,rho,ef);

    ef += 1.0;

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    efhat(0) = complex<double>(0.0,0.0);

    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index ii = 1; ii < (Nx/2+1); ii++){
      complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*ii);

      efhat(ii) /= (lambdax/ncx);
    }

    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhat.begin(),ef.begin());

    // Here I have the electric field at time tau/2, so restart integration

    /* Half step K */

    tmpX = lr_sol.X;
    matmul(tmpX,lr_sol.S,lr_sol.X);

    fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    matmul(Khat,Tc,Mhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row(lr_sol.X,ef.begin(),lr_sol.X);

      fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Kehat.begin());

      matmul_transb(Kehat,C2c,Khat);

      matmul(Khat,Tc,Mhattmp);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

          Mhat(j,k) *= exp(-tau_ee_h*lambdax*dc_r(k));
          Mhat(j,k) += tau_ee_h*phi1_im(-tau_ee_h*lambdax*dc_r(k))*Mhattmp(j,k);
        }
      }

      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

      // Second stage

      ptw_mult_row(lr_sol.X,ef.begin(),lr_sol.X);

      fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Kehat.begin());

      matmul_transb(Kehat,C2c,Khat);

      matmul(Khat,Tc,Kehat);

      Kehat -= Mhattmp;

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

          Mhat(j,k) += tau_ee_h*phi2_im(-tau_ee_h*lambdax*dc_r(k))*Kehat(j,k);
        }
      }

      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_x);

    /* Half step S */

    #ifdef __OPENMP__
    #pragma omp parallel for
    #endif
    for(Index ii = 0; ii < Nx; ii++){
      wx(ii) = hx*ef(ii);
    }

    coeff(lr_sol.X, lr_sol.X, wx.begin(), D1);

    fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

    ptw_mult_row(Khat,lambdax_n.begin(),Khat);

    fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),dX.begin());

    coeff(lr_sol.X, dX, hx, D2);

    // RK4
    for(int j = 0; j< nsteps_rk4; j++){
      matmul_transb(lr_sol.S,C1,tmpS);
      matmul(D2,tmpS,tmpS1);

      matmul_transb(lr_sol.S,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS1 -= multmp;

      T = tmpS1;
      T *= (tau_rk4_h/2);
      T += lr_sol.S;

      matmul_transb(T,C1,tmpS);
      matmul(D2,tmpS,tmpS2);

      matmul_transb(T,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS2 -= multmp;

      T = tmpS2;
      T *= (tau_rk4_h/2);
      T += lr_sol.S;

      matmul_transb(T,C1,tmpS);
      matmul(D2,tmpS,tmpS3);

      matmul_transb(T,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS3 -= multmp;

      T = tmpS3;
      T *= tau_rk4_h;
      T += lr_sol.S;

      matmul_transb(T,C1,tmpS);
      matmul(D2,tmpS,tmpS4);

      matmul_transb(T,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS4 -= multmp;

      tmpS2 *= 2.0;
      tmpS3 *= 2.0;

      tmpS1 += tmpS2;
      tmpS1 += tmpS3;
      tmpS1 += tmpS4;
      tmpS1 *= (tau_rk4_h/6.0);

      lr_sol.S += tmpS1;
    }

    /* Full step L */

    tmpV = lr_sol.V;
    matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    schur(D1, W, dd_r, lwork);

    W.to_cplx(Wc);
    D2.to_cplx(D2c);

    fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

    matmul(Lhat,Wc,Nhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row(lr_sol.V,v.begin(),lr_sol.V);

      fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lvhat.begin());

      matmul_transb(Lvhat,D2c,Lhat);

      matmul(Lhat,Wc,Nhattmp);

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nv/2 + 1); j++){
          complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(bv-av)*j);

          Nhat(j,k) *= exp(tau_ee*lambdav*dd_r(k));
          Nhat(j,k) += -tau_ee*phi1_im(tau_ee*lambdav*dd_r(k))*Nhattmp(j,k);
        }
      }

      matmul_transb(Nhat,Wc,Lhat);

      Lhat *= ncv;

      fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      // Second stage

      ptw_mult_row(lr_sol.V,v.begin(),lr_sol.V);

      fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lvhat.begin());

      matmul_transb(Lvhat,D2c,Lhat);

      matmul(Lhat,Wc,Lvhat);

      Lvhat -= Nhattmp;

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nv/2 + 1); j++){
          complex<double> lambdav = complex<double>(0.0,2.0*M_PI/(bv-av)*j);

          Nhat(j,k) += -tau_ee*phi2_im(tau_ee*lambdav*dd_r(k))*Lvhat(j,k);
        }
      }

      matmul_transb(Nhat,Wc,Lhat);

      Lhat *= ncv;

      fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

    }

    gram_schmidt(lr_sol.V, lr_sol.S, ip_v);
    transpose_inplace(lr_sol.S);

    /* Half step S */

    coeff(lr_sol.V, lr_sol.V, wv.begin(), C1);

    fftw_execute_dft_r2c(plans_v[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

    ptw_mult_row(Lhat,lambdav_n.begin(),Lhat);

    fftw_execute_dft_c2r(plans_v[1],(fftw_complex*)Lhat.begin(),dV.begin());

    coeff(lr_sol.V, dV, hv, C2);

    // RK4
    for(int j = 0; j< nsteps_rk4; j++){
      matmul_transb(lr_sol.S,C1,tmpS);
      matmul(D2,tmpS,tmpS1);

      matmul_transb(lr_sol.S,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS1 -= multmp;

      T = tmpS1;
      T *= (tau_rk4_h/2);
      T += lr_sol.S;

      matmul_transb(T,C1,tmpS);
      matmul(D2,tmpS,tmpS2);

      matmul_transb(T,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS2 -= multmp;

      T = tmpS2;
      T *= (tau_rk4_h/2);
      T += lr_sol.S;

      matmul_transb(T,C1,tmpS);
      matmul(D2,tmpS,tmpS3);

      matmul_transb(T,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS3 -= multmp;

      T = tmpS3;
      T *= tau_rk4_h;
      T += lr_sol.S;

      matmul_transb(T,C1,tmpS);
      matmul(D2,tmpS,tmpS4);

      matmul_transb(T,C2,tmpS);
      matmul(D1,tmpS,multmp);

      tmpS4 -= multmp;

      tmpS2 *= 2.0;
      tmpS3 *= 2.0;

      tmpS1 += tmpS2;
      tmpS1 += tmpS3;
      tmpS1 += tmpS4;
      tmpS1 *= (tau_rk4_h/6.0);

      lr_sol.S += tmpS1;
    }

    /* Half step K */

    tmpX = lr_sol.X;
    matmul(tmpX,lr_sol.S,lr_sol.X);

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

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

          Mhat(j,k) *= exp(-tau_ee_h*lambdax*dc_r(k));
          Mhat(j,k) += tau_ee_h*phi1_im(-tau_ee_h*lambdax*dc_r(k))*Mhattmp(j,k);
        }
      }

      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

      // Second stage

      ptw_mult_row(lr_sol.X,ef.begin(),lr_sol.X);

      fftw_execute_dft_r2c(plans_x[0],lr_sol.X.begin(),(fftw_complex*)Kehat.begin());

      matmul_transb(Kehat,C2c,Khat);

      matmul(Khat,Tc,Kehat);

      Kehat -= Mhattmp;

      #ifdef __OPENMP__
      #pragma omp parallel for collapse(2)
      #endif
      for(int k = 0; k < r; k++){
        for(Index j = 0; j < (Nx/2 + 1); j++){
          complex<double> lambdax = complex<double>(0.0,2.0*M_PI/(bx-ax)*j);

          Mhat(j,k) += tau_ee_h*phi2_im(-tau_ee_h*lambdax*dc_r(k))*Kehat(j,k);
        }
      }

      matmul_transb(Mhat,Tc,Khat);

      Khat *= ncx;

      fftw_execute_dft_c2r(plans_x[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

    }

    gram_schmidt(lr_sol.X, lr_sol.S, ip_x);

    gt::stop("Restarted integration CPU");

    // Error mass
    coeff_one(lr_sol.X,hx,int_x);

    coeff_one(lr_sol.V,hv,int_v);

    matvec(lr_sol.S,int_v,tmp_vec);

    mass = 0.0;
    for(int ii = 0; ii < r; ii++){
      mass += (int_x(ii)*tmp_vec(ii));
    }

    err_mass = abs(mass0-mass);

    err_massf << err_mass << endl;

    coeff_one(lr_sol.V,wv2,int_v);

    matvec(lr_sol.S,int_v,tmp_vec);

    energy = el_energy;
    for(int ii = 0; ii < r; ii++){
      energy += (0.5*int_x(ii)*tmp_vec(ii));
    }

    err_energy = abs(energy0-energy);

    err_energyf << err_energy << endl;
    #endif

    // GPU
    #ifdef __CUDACC__

    /* Lie Splitting to obtain the electric field */

    d_lr_sol_e.X = d_lr_sol.X;
    d_lr_sol_e.S = d_lr_sol.S;
    d_lr_sol_e.V = d_lr_sol.V;

    /* Half step K */

    d_tmpX = d_lr_sol_e.X;

    matmul(d_tmpX,d_lr_sol_e.S,d_lr_sol_e.X);

    gt::start("Lie splitting for electric field GPU");

    // Electric field

    coeff_one(d_lr_sol_e.V,-hv,d_rho);

    matvec(d_lr_sol_e.X,d_rho,d_ef);

    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier<<<(Nx/2+1+n_threads-1)/n_threads,n_threads>>>(Nx/2+1, d_efhat.begin(), ax, bx, Nx); //128 threads blocks as needed

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhat.begin(),d_ef.begin());

    // Electric energy

    cublasDdot (handle_dot, Nx, d_ef.begin(), 1, d_ef.begin(), 1, d_el_energy);
    cudaDeviceSynchronize();
    scale_unique<<<1,1>>>(d_el_energy,0.5*hx); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_el_energy_CPU,d_el_energy,sizeof(double),cudaMemcpyDeviceToHost);

    el_energyGPUf << d_el_energy_CPU << endl;


    // Main of K step
    coeff(d_lr_sol_e.V, d_lr_sol_e.V, d_wv.begin(), d_C1);

    cufftExecD2Z(plans_d_v[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

    ptw_mult_row_cplx_fourier<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.shape()[0], Nv, d_Lhat.begin(), av, bv);

    cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_dV.begin());

    coeff(d_lr_sol_e.V, d_dV, hv, d_C2);

    C1_gpu = d_C1;
    schur(C1_gpu, T_gpu, dc_r_gpu, lwork);
    d_T = T_gpu;
    d_dc_r = dc_r_gpu;

    cplx_conv<<<(d_T.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_T.num_elements(), d_T.begin(), d_Tc.begin());

    cplx_conv<<<(d_C2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2.num_elements(), d_C2.begin(), d_C2c.begin());

    cufftExecD2Z(plans_d_x[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

    matmul(d_Khat,d_Tc,d_Mhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_ef.begin(),d_lr_sol_e.X.begin());

      cufftExecD2Z(plans_d_x[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_Kehat.begin());

      matmul_transb(d_Kehat,d_C2c,d_Khat);

      matmul(d_Khat,d_Tc,d_Mhattmp);

      exp_euler_fourier<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), d_Mhat.shape()[0], d_Mhat.begin(), d_dc_r.begin(), tau_ee_h, d_Mhattmp.begin(), ax, bx);

      matmul_transb(d_Mhat,d_Tc,d_Khat);

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(),d_Khat.begin(),ncx);

      cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol_e.X.begin());

      // Second stage

      ptw_mult_row_k<<<(d_lr_sol_e.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.X.num_elements(),d_lr_sol_e.X.shape()[0],d_lr_sol_e.X.begin(),d_ef.begin(),d_lr_sol_e.X.begin());

      cufftExecD2Z(plans_d_x[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_Kehat.begin());

      matmul_transb(d_Kehat,d_C2c,d_Khat);

      matmul(d_Khat,d_Tc,d_Kehat);

      second_ord_stage_fourier<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), d_Mhat.shape()[0], d_Mhat.begin(), d_dc_r.begin(), tau_ee_h, d_Mhattmp.begin(), d_Kehat.begin(), ax, bx);

      matmul_transb(d_Mhat,d_Tc,d_Khat);

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(),d_Khat.begin(),ncx);

      cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol_e.X.begin());

    }

    gram_schmidt_gpu(d_lr_sol_e.X, d_lr_sol_e.S, hx, gen);

    /* Half step S */

    ptw_mult_scal<<<(d_ef.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_ef.num_elements(), d_ef.begin(), hx, d_wx.begin());

    coeff(d_lr_sol_e.X, d_lr_sol_e.X, d_wx.begin(), d_D1);

    cufftExecD2Z(plans_d_x[0],d_lr_sol_e.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

    ptw_mult_row_cplx_fourier<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.shape()[0], Nx, d_Khat.begin(), ax, bx);

    cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_dX.begin());

    coeff(d_lr_sol_e.X, d_dX, hx, d_D2);

    // RK4
    for(int j = 0; j< nsteps_rk4; j++){
      matmul_transb(d_lr_sol_e.S,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS1);

      matmul_transb(d_lr_sol_e.S,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4_h/2.0, d_lr_sol_e.S.begin(), d_tmpS1.begin(),d_multmp.begin());

      matmul_transb(d_T,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS2);

      matmul_transb(d_T,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4_h/2.0, d_lr_sol_e.S.begin(), d_tmpS2.begin(),d_multmp.begin());

      matmul_transb(d_T,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS3);

      matmul_transb(d_T,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4_h, d_lr_sol_e.S.begin(), d_tmpS3.begin(),d_multmp.begin());

      matmul_transb(d_T,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS4);

      matmul_transb(d_T,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4_finalcomb<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_lr_sol_e.S.begin(), tau_rk4_h, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_multmp.begin());
    }

    /* Half step L */
    d_tmpV = d_lr_sol_e.V;

    matmul_transb(d_tmpV,d_lr_sol_e.S,d_lr_sol_e.V);

    D1_gpu = d_D1;
    schur(D1_gpu, W_gpu, dd_r_gpu, lwork);
    d_W = W_gpu;
    d_dd_r = dd_r_gpu;

    cplx_conv<<<(d_T.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_W.num_elements(), d_W.begin(), d_Wc.begin());

    cplx_conv<<<(d_D2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2.num_elements(), d_D2.begin(), d_D2c.begin());

    cufftExecD2Z(plans_d_v[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

    matmul(d_Lhat,d_Wc,d_Nhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_v.begin(),d_lr_sol_e.V.begin());

      cufftExecD2Z(plans_d_v[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_Lvhat.begin());

      matmul_transb(d_Lvhat,d_D2c,d_Lhat);

      matmul(d_Lhat,d_Wc,d_Nhattmp);

      exp_euler_fourier<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), d_Nhat.shape()[0], d_Nhat.begin(), d_dd_r.begin(), -tau_ee_h, d_Nhattmp.begin(), av, bv);

      matmul_transb(d_Nhat,d_Wc,d_Lhat);

      ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(),d_Lhat.begin(),ncv);

      cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol_e.V.begin());

      // Second stage

      ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol_e.V.num_elements(),d_lr_sol_e.V.shape()[0],d_lr_sol_e.V.begin(),d_v.begin(),d_lr_sol_e.V.begin());

      cufftExecD2Z(plans_d_v[0],d_lr_sol_e.V.begin(),(cufftDoubleComplex*)d_Lvhat.begin());

      matmul_transb(d_Lvhat,d_D2c,d_Lhat);

      matmul(d_Lhat,d_Wc,d_Lvhat);

      second_ord_stage_fourier<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), d_Nhat.shape()[0], d_Nhat.begin(), d_dd_r.begin(), -tau_ee_h, d_Nhattmp.begin(), d_Lvhat.begin(), av, bv);

      matmul_transb(d_Nhat,d_Wc,d_Lhat);

      ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(),d_Lhat.begin(),ncv);

      cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol_e.V.begin());

    }

    cudaDeviceSynchronize();
    gt::stop("Lie splitting for electric field GPU");

    gt::start("Restarted integration GPU");

    // Electric field at tau/2

    coeff_one(d_lr_sol_e.V,-hv,d_rho);

    matvec(d_lr_sol_e.X,d_rho,d_ef);

    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier<<<(Nx/2+1+n_threads-1)/n_threads,n_threads>>>(Nx/2+1, d_efhat.begin(), ax, bx, Nx); //128 threads blocks as needed

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhat.begin(),d_ef.begin());

    // Here I have the electric field at time tau/2, so restart integration

    /* Half step K */

    d_tmpX = d_lr_sol.X;

    matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

    matmul(d_Khat,d_Tc,d_Mhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_ef.begin(),d_lr_sol.X.begin());

      cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Kehat.begin());

      matmul_transb(d_Kehat,d_C2c,d_Khat);

      matmul(d_Khat,d_Tc,d_Mhattmp);

      exp_euler_fourier<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), d_Mhat.shape()[0], d_Mhat.begin(), d_dc_r.begin(), tau_ee_h, d_Mhattmp.begin(), ax, bx);

      matmul_transb(d_Mhat,d_Tc,d_Khat);

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(),d_Khat.begin(),ncx);

      cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      // Second stage

      ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_ef.begin(),d_lr_sol.X.begin());

      cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Kehat.begin());

      matmul_transb(d_Kehat,d_C2c,d_Khat);

      matmul(d_Khat,d_Tc,d_Kehat);

      second_ord_stage_fourier<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), d_Mhat.shape()[0], d_Mhat.begin(), d_dc_r.begin(), tau_ee_h, d_Mhattmp.begin(), d_Kehat.begin(), ax, bx);

      matmul_transb(d_Mhat,d_Tc,d_Khat);

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(),d_Khat.begin(),ncx);

      cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

    }

    gram_schmidt_gpu(d_lr_sol.X, d_lr_sol.S, hx, gen);

    /* Half step S */

    ptw_mult_scal<<<(d_ef.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_ef.num_elements(), d_ef.begin(), hx, d_wx.begin());

    coeff(d_lr_sol.X, d_lr_sol.X, d_wx.begin(), d_D1);

    cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

    ptw_mult_row_cplx_fourier<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.shape()[0], Nx, d_Khat.begin(), ax, bx);

    cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_dX.begin());

    coeff(d_lr_sol.X, d_dX, hx, d_D2);

    // RK4
    for(int j = 0; j< nsteps_rk4; j++){
      matmul_transb(d_lr_sol.S,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS1);

      matmul_transb(d_lr_sol.S,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS1.begin(),d_multmp.begin());

      matmul_transb(d_T,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS2);

      matmul_transb(d_T,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS2.begin(),d_multmp.begin());

      matmul_transb(d_T,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS3);

      matmul_transb(d_T,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4_h, d_lr_sol.S.begin(), d_tmpS3.begin(),d_multmp.begin());

      matmul_transb(d_T,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS4);

      matmul_transb(d_T,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4_finalcomb<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_lr_sol.S.begin(), tau_rk4_h, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_multmp.begin());
    }

    /* Full step L */
    d_tmpV = d_lr_sol.V;

    matmul_transb(d_tmpV,d_lr_sol.S,d_lr_sol.V);

    D1_gpu = d_D1;
    schur(D1_gpu, W_gpu, dd_r_gpu, lwork);
    d_W = W_gpu;
    d_dd_r = dd_r_gpu;

    cplx_conv<<<(d_T.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_W.num_elements(), d_W.begin(), d_Wc.begin());
    cplx_conv<<<(d_D2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2.num_elements(), d_D2.begin(), d_D2c.begin());

    cufftExecD2Z(plans_d_v[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

    matmul(d_Lhat,d_Wc,d_Nhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_lr_sol.V.begin());

      cufftExecD2Z(plans_d_v[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lvhat.begin());

      matmul_transb(d_Lvhat,d_D2c,d_Lhat);

      matmul(d_Lhat,d_Wc,d_Nhattmp);

      exp_euler_fourier<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), d_Nhat.shape()[0], d_Nhat.begin(), d_dd_r.begin(), -tau_ee, d_Nhattmp.begin(), av, bv);

      matmul_transb(d_Nhat,d_Wc,d_Lhat);

      ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(),d_Lhat.begin(),ncv);

      cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      // Second stage

      ptw_mult_row_k<<<(d_lr_sol_e.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_lr_sol.V.begin());

      cufftExecD2Z(plans_d_v[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lvhat.begin());

      matmul_transb(d_Lvhat,d_D2c,d_Lhat);

      matmul(d_Lhat,d_Wc,d_Lvhat);

      second_ord_stage_fourier<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), d_Nhat.shape()[0], d_Nhat.begin(), d_dd_r.begin(), -tau_ee, d_Nhattmp.begin(), d_Lvhat.begin(), av, bv);

      matmul_transb(d_Nhat,d_Wc,d_Lhat);

      ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(),d_Lhat.begin(),ncv);

      cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

    }

    gram_schmidt_gpu(d_lr_sol.V, d_lr_sol.S, hv, gen);
    transpose_inplace<<<d_lr_sol.S.num_elements(),1>>>(r,d_lr_sol.S.begin());

    /* Half step S */

    coeff(d_lr_sol.V, d_lr_sol.V, d_wv.begin(), d_C1);

    cufftExecD2Z(plans_d_v[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

    ptw_mult_row_cplx_fourier<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.shape()[0], Nv, d_Lhat.begin(), av, bv);

    cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_dV.begin());

    coeff(d_lr_sol.V, d_dV, hv, d_C2);

    // RK4
    for(int j = 0; j< nsteps_rk4; j++){
      matmul_transb(d_lr_sol.S,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS1);

      matmul_transb(d_lr_sol.S,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS1.begin(),d_multmp.begin());

      matmul_transb(d_T,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS2);

      matmul_transb(d_T,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4_h/2.0, d_lr_sol.S.begin(), d_tmpS2.begin(),d_multmp.begin());

      matmul_transb(d_T,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS3);

      matmul_transb(d_T,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_T.begin(), tau_rk4_h, d_lr_sol.S.begin(), d_tmpS3.begin(),d_multmp.begin());

      matmul_transb(d_T,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_tmpS4);

      matmul_transb(d_T,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      rk4_finalcomb<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_lr_sol.S.begin(), tau_rk4_h, d_tmpS1.begin(), d_tmpS2.begin(), d_tmpS3.begin(), d_tmpS4.begin(),d_multmp.begin());
    }

    /* Half step K */

    d_tmpX = d_lr_sol.X;
    matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    C1_gpu = d_C1;
    schur(C1_gpu, T_gpu, dc_r_gpu, lwork);
    d_T = T_gpu;
    d_dc_r = dc_r_gpu;

    cplx_conv<<<(d_T.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_T.num_elements(), d_T.begin(), d_Tc.begin());
    cplx_conv<<<(d_C2.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_C2.num_elements(), d_C2.begin(), d_C2c.begin());

    cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

    matmul(d_Khat,d_Tc,d_Mhat);

    for(int kk = 0; kk < nsteps_ee; kk++){

      ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_ef.begin(),d_lr_sol.X.begin());

      cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Kehat.begin());

      matmul_transb(d_Kehat,d_C2c,d_Khat);

      matmul(d_Khat,d_Tc,d_Mhattmp);

      exp_euler_fourier<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), d_Mhat.shape()[0], d_Mhat.begin(), d_dc_r.begin(), tau_ee_h, d_Mhattmp.begin(), ax, bx);

      matmul_transb(d_Mhat,d_Tc,d_Khat);

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(),d_Khat.begin(),ncx);

      cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      // Second stage

      ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_ef.begin(),d_lr_sol.X.begin());

      cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Kehat.begin());

      matmul_transb(d_Kehat,d_C2c,d_Khat);

      matmul(d_Khat,d_Tc,d_Kehat);

      second_ord_stage_fourier<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), d_Mhat.shape()[0], d_Mhat.begin(), d_dc_r.begin(), tau_ee_h, d_Mhattmp.begin(), d_Kehat.begin(), ax, bx);

      matmul_transb(d_Mhat,d_Tc,d_Khat);

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(),d_Khat.begin(),ncx);

      cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

    }

    gram_schmidt_gpu(d_lr_sol.X, d_lr_sol.S, hx, gen);

    cudaDeviceSynchronize();
    gt::stop("Restarted integration GPU");

    // Error mass
    coeff_one(d_lr_sol.X,hx,d_int_x);

    coeff_one(d_lr_sol.V,hv,d_int_v);

    matvec(d_lr_sol.S,d_int_v,d_tmp_vec);

    cublasDdot (handle_dot, r, d_int_x.begin(), 1, d_tmp_vec.begin(), 1,d_mass);
    cudaDeviceSynchronize();
    cudaMemcpy(&d_mass_CPU,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    err_mass_CPU = abs(mass0-d_mass_CPU);

    err_massGPUf << err_mass_CPU << endl;

    coeff_one(d_lr_sol.V,d_wv2,d_int_v);

    matvec(d_lr_sol.S,d_int_v,d_tmp_vec);

    cublasDdot (handle_dot, r, d_int_x.begin(), 1, d_tmp_vec.begin(), 1,d_energy);
    cudaDeviceSynchronize();
    scale_unique<<<1,1>>>(d_energy,0.5); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_energy_CPU,d_energy,sizeof(double),cudaMemcpyDeviceToHost);

    err_energy_CPU = abs(energy0-(d_energy_CPU+d_el_energy_CPU));

    err_energyGPUf << err_energy_CPU << endl;

    #endif
    #ifdef __CPU__
    cout << "Electric energy CPU: " << el_energy << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Electric energy GPU: " << d_el_energy_CPU << endl;
    #endif
    #ifdef __CPU__
    cout << "Error in mass CPU: " << err_mass << endl;
    #endif
    #ifdef __CUDACC__
    cout << "Error in mass GPU: " << err_mass_CPU << endl;
    #endif
    #ifdef __CPU__
    cout << "Error in energy CPU: " << err_energy << endl;
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
  destroy_plans(plans_d_x);
  destroy_plans(plans_d_v);

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

  Index Nx = 32; // NEEDS TO BE EVEN FOR FOURIER
  Index Nv = 32; // NEEDS TO BE EVEN FOR FOURIER

  int r = 10; // rank desired

  double tstar = 10.0; // final time

  Index nsteps_ref = 10000;

  vector<Index> nspan = {1800,2200,2600,3000,3400};

  int nsteps_ee = 1; // number of time steps for exponential integrator
  int nsteps_rk4 = 1; // number of time steps for rk4

/*
  // Linear Landau
  double ax = 0;
  double bx = 4.0*M_PI;

  double av = -6.0;
  double bv = 6.0;

  double alpha = 0.01;
  double kappa = 0.5;
*/

  // Two stream instability

  double ax = 0;
  double bx = 10.0*M_PI;

  double av = -9.0;
  double bv = 9.0;

  double alpha = 0.001;
  double kappa = 1.0/5.0;
  double v0 = 2.4;

  bool ee_flag = false;

  // Initial datum generation

  double hx = (bx-ax) / Nx;
  double hv = (bv-av) / Nv;

  vector<double*> X, V;

  multi_array<double,1> xx({Nx});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index i = 0; i < Nx; i++){
    xx(i) = 1.0 + alpha*cos(kappa*(ax + i*hx));
  }
  X.push_back(xx.begin());

  multi_array<double,1> vv({Nv});

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index i = 0; i < Nv; i++){
    //vv(i) = (1.0/sqrt(2*M_PI)) *exp(-pow((av + i*hv),2)/2.0);
    vv(i) = (1.0/sqrt(8*M_PI)) * (exp(-pow((av + i*hv - v0),2)/2.0)+exp(-pow((av + i*hv +v0),2)/2.0));
  }
  V.push_back(vv.begin());

  lr2<double> lr_sol0(r,{Nx,Nv});

  std::function<double(double*,double*)> ip_x = inner_product_from_const_weight(hx, Nx);
  std::function<double(double*,double*)> ip_v = inner_product_from_const_weight(hv, Nv);

  // Mandatory to initialize after plan creation if we use FFTW_MEASURE
  multi_array<double,1> ef({Nx});
  multi_array<complex<double>,1> efhat({Nx/2 + 1});

  array<fftw_plan,2> plans_e = create_plans_1d(Nx, ef, efhat);

  multi_array<complex<double>,2> Khat({Nx/2+1,r});
  multi_array<complex<double>,2> Lhat({Nv/2+1,r});

  array<fftw_plan,2> plans_x = create_plans_1d(Nx, lr_sol0.X, Khat);
  array<fftw_plan,2> plans_v = create_plans_1d(Nv, lr_sol0.V, Lhat);

  initialize(lr_sol0, X, V, ip_x, ip_v);

  // Computation of reference solution
  lr2<double> lr_sol_fin(r,{Nx,Nv});

  //cout << "First order" << endl;
  //lr_sol_fin = integration_first_order(Nx,Nv,r,tstar,nsteps_ref,nsteps_ee,nsteps_rk4,ax,bx,av,bv,alpha,kappa,ee_flag,lr_sol0, plans_e, plans_x, plans_v);

  cout << "Second order" << endl;
  lr_sol_fin = integration_second_order(Nx,Nv,r,tstar,nsteps_ref,nsteps_ee,nsteps_rk4,ax,bx,av,bv,alpha,kappa,lr_sol0, plans_e, plans_x, plans_v);

  //exit(1);

  //cout << gt::sorted_output() << endl;

  multi_array<double,2> refsol({Nx,Nv});
  multi_array<double,2> sol({Nx,Nv});
  multi_array<double,2> tmpsol({Nx,r});

  matmul(lr_sol_fin.X,lr_sol_fin.S,tmpsol);
  matmul_transb(tmpsol,lr_sol_fin.V,refsol);

  ofstream error_order1_1d;
  error_order1_1d.open("../../plots/error_order1_1d.txt");
  error_order1_1d.precision(16);

  ofstream error_order2_1d;
  error_order2_1d.open("../../plots/error_order2_1d.txt");
  error_order2_1d.precision(16);

  error_order1_1d << nspan.size() << endl;
  for(int count = 0; count < nspan.size(); count++){
    error_order1_1d << nspan[count] << endl;
  }

  error_order2_1d << nspan.size() << endl;
  for(int count = 0; count < nspan.size(); count++){
    error_order2_1d << nspan[count] << endl;
  }

  for(int count = 0; count < nspan.size(); count++){

    lr_sol_fin = integration_first_order(Nx,Nv,r,tstar,nspan[count],nsteps_ee,nsteps_rk4,ax,bx,av,bv,alpha,kappa,ee_flag,lr_sol0, plans_e, plans_x, plans_v);

    matmul(lr_sol_fin.X,lr_sol_fin.S,tmpsol);
    matmul_transb(tmpsol,lr_sol_fin.V,sol);

    double error_o1 = 0.0;
    for(int iii = 0; iii < Nx; iii++){
      for(int jjj = 0; jjj < Nv; jjj++){
        double value = abs(refsol(iii,jjj)-sol(iii,jjj));
        if( error_o1 < value){
          error_o1 = value;
        }
      }
    }
    error_order1_1d << error_o1 << endl;

    lr_sol_fin = integration_second_order(Nx,Nv,r,tstar,nspan[count],nsteps_ee,nsteps_rk4,ax,bx,av,bv,alpha,kappa,lr_sol0, plans_e, plans_x, plans_v);

    matmul(lr_sol_fin.X,lr_sol_fin.S,tmpsol);
    matmul_transb(tmpsol,lr_sol_fin.V,sol);


    double error_o2 = 0.0;
    for(int iii = 0; iii < Nx; iii++){
      for(int jjj = 0; jjj < Nv; jjj++){
        double value = abs(refsol(iii,jjj)-sol(iii,jjj));
        if( error_o2 < value){
          error_o2 = value;
        }
      }
    }

    error_order2_1d << error_o2 << endl;

  }

  error_order1_1d.close();
  error_order2_1d.close();

  destroy_plans(plans_e);
  destroy_plans(plans_x);
  destroy_plans(plans_v);


  return 0;
}
