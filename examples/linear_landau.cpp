#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/kernels.hpp>

#ifdef __CUDACC__
  cublasHandle_t  handle;
#endif


int main(){

  Index Nx = 64; // NEEDS TO BE EVEN FOR FOURIER
  Index Nv = 256; // NEEDS TO BE EVEN FOR FOURIER

  int r = 5; // rank desired

  double tstar = 100; // final time
  double tau = 0.00625; // time step splitting

  int nsteps_ee = 1; // number of time steps for explicit euler

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

  ef += 1.0;

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

  coeff_one(lr_sol.V,wv2,int_v);

  matvec(lr_sol.S,int_v,tmp_vec);

  for(int ii = 0; ii < r; ii++){
    energy0 += (0.5*int_x(ii)*tmp_vec(ii));
  }

  cout.precision(5);
  cout << std::scientific;

  //ofstream el_energyf;
  //ofstream err_massf;
  //ofstream err_energyf;

  //el_energyf.open("el_energy_1d.txt");
  //err_massf.open("err_mass_1d.txt");
  //err_energyf.open("err_energy_1d.txt");

  //el_energyf.precision(16);
  //err_massf.precision(16);
  //err_energyf.precision(16);

  //el_energyf << tstar << endl;
  //el_energyf << tau << endl;

  multi_array<double,2> tmpX({Nx,r});
  multi_array<double,2> tmpV({Nv,r});
  multi_array<double,2> tmpS({r,r});

  //// FOR GPU ////
  #ifdef __CUDACC__
  cublasCreate (&handle);

  // To be substituted if we initialize in GPU

  lr2<double> d_lr_sol(r,{Nx,Nv}, stloc::device);
  d_lr_sol.X = lr_sol.X;
  d_lr_sol.S = lr_sol.S;
  d_lr_sol.V = lr_sol.V;

  // Electric field
  multi_array<double,1> d_rho({r},stloc::device);
  multi_array<double,1> d_ef({Nx},stloc::device);
  multi_array<cuDoubleComplex,1> d_efhat({Nx/2 + 1},stloc::device);

  array<cufftHandle,2> plans_d_e = create_plans_1d(Nx);

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

  //ofstream el_energyGPUf;
  //ofstream err_massGPUf;
  //ofstream err_energyGPUf;

  //el_energyGPUf.open("el_energy_gpu_1d.txt");
  //err_massGPUf.open("err_mass_gpu_1d.txt");
  //err_energyGPUf.open("err_energy_gpu_1d.txt");

  //el_energyGPUf.precision(16);
  //err_massGPUf.precision(16);
  //err_energyGPUf.precision(16);

  //el_energyGPUf << tstar << endl;
  //el_energyGPUf << tau << endl;

  // Temporary
  multi_array<double,2> d_tmpX({Nx,r},stloc::device);
  multi_array<double,2> d_tmpS({r,r},stloc::device);
  multi_array<double,2> d_tmpV({Nv,r},stloc::device);
  #endif

  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i << " on " << nsteps << endl;

    // CPU
    /* K step */
    tmpX = lr_sol.X;

    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field

    coeff_one(lr_sol.V,-hv,rho);

    matvec(lr_sol.X,rho,ef);

    ef += 1.0;

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

    //err_massf << err_mass << endl;

    coeff_one(lr_sol.V,wv2,int_v);

    matvec(lr_sol.S,int_v,tmp_vec);

    energy = el_energy;
    for(int ii = 0; ii < r; ii++){
      energy += (0.5*int_x(ii)*tmp_vec(ii));
    }

    err_energy = abs(energy0-energy);

    //err_energyf << err_energy << endl;

    // GPU
  #ifdef __CUDACC__
    // K step

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

      exp_euler_fourier<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), d_Mhat.shape()[0], d_Mhat.begin(), d_dc_r.begin(), ts_ee, d_Mhattmp.begin(), ax, bx);

      matmul_transb(d_Mhat,d_Tc,d_Khat);

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(),d_Khat.begin(),ncx);

      cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

    }

    gram_schmidt_gpu(d_lr_sol.X, d_lr_sol.S, hx);

    // S step

    ptw_mult_scal<<<(d_ef.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_ef.num_elements(), d_ef.begin(), hx, d_wx.begin());

    coeff(d_lr_sol.X, d_lr_sol.X, d_wx.begin(), d_D1);

    cufftExecD2Z(plans_d_x[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

    ptw_mult_row_cplx_fourier<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.shape()[0], Nx, d_Khat.begin(), ax, bx);

    cufftExecZ2D(plans_d_x[1],(cufftDoubleComplex*)d_Khat.begin(),d_dX.begin());

    coeff(d_lr_sol.X, d_dX, hx, d_D2);

    // Explicit Euler
    for(int j = 0; j< nsteps_ee; j++){
      matmul_transb(d_lr_sol.S,d_C1,d_tmpS);
      matmul(d_D2,d_tmpS,d_T);

      matmul_transb(d_lr_sol.S,d_C2,d_tmpS);
      matmul(d_D1,d_tmpS,d_multmp);

      expl_euler<<<(d_D1.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D1.num_elements(), d_lr_sol.S.begin(), ts_ee, d_T.begin(), d_multmp.begin());
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

      exp_euler_fourier<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), d_Nhat.shape()[0], d_Nhat.begin(), d_dc_r.begin(), -ts_ee, d_Nhattmp.begin(), av, bv);

      matmul_transb(d_Nhat,d_Tc,d_Lhat);

      ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(),d_Lhat.begin(),ncv);

      cufftExecZ2D(plans_d_v[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

    }

    gram_schmidt_gpu(d_lr_sol.V, d_lr_sol.S, hv);

    transpose_inplace<<<d_lr_sol.S.num_elements(),1>>>(r,d_lr_sol.S.begin());

    cublasDdot (handle, Nx, d_ef.begin(), 1, d_ef.begin(), 1, d_el_energy);
    scale_unique<<<1,1>>>(d_el_energy,0.5*hx); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_el_energy_CPU,d_el_energy,sizeof(double),cudaMemcpyDeviceToHost);

    //el_energyGPUf << d_el_energy_CPU << endl;

    // Error mass
    coeff_one(d_lr_sol.X,hx,d_int_x);

    coeff_one(d_lr_sol.V,hv,d_int_v);

    matvec(d_lr_sol.S,d_int_v,d_tmp_vec);

    cublasDdot (handle, r, d_int_x.begin(), 1, d_tmp_vec.begin(), 1,d_mass);

    cudaMemcpy(&d_mass_CPU,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    err_mass_CPU = abs(mass0-d_mass_CPU);

    //err_massGPUf << err_mass_CPU << endl;

    coeff_one(d_lr_sol.V,d_wv2,d_int_v);

    matvec(d_lr_sol.S,d_int_v,d_tmp_vec);

    cublasDdot (handle, r, d_int_x.begin(), 1, d_tmp_vec.begin(), 1,d_energy);
    scale_unique<<<1,1>>>(d_energy,0.5); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_energy_CPU,d_energy,sizeof(double),cudaMemcpyDeviceToHost);

    err_energy_CPU = abs(energy0-(d_energy_CPU+d_el_energy_CPU));

    //err_energyGPUf << err_energy_CPU << endl;

    #endif

    cout << "Electric energy CPU: " << el_energy << endl;
    #ifdef __CUDACC__
    cout << "Electric energy GPU: " << d_el_energy_CPU << endl;
    #endif

    cout << "Error in mass CPU: " << err_mass << endl;
    #ifdef __CUDACC__
    cout << "Error in mass GPU: " << err_mass_CPU << endl;
    #endif

    cout << "Error in energy CPU: " << err_energy << endl;
    #ifdef __CUDACC__
    cout << "Error in energy GPU: " << err_energy_CPU << endl;
    #endif

  }

  destroy_plans(plans_e);
  destroy_plans(plans_x);
  destroy_plans(plans_v);

  //el_energyf.close();
  //err_massf.close();
  //err_energyf.close();

  #ifdef __CUDACC__
  cublasDestroy(handle);

  destroy_plans(plans_d_e);
  destroy_plans(plans_d_x);
  destroy_plans(plans_d_v);

  //el_energyGPUf.close();
  //err_massGPUf.close();
  //err_energyGPUf.close();
  #endif

  return 0;
}
