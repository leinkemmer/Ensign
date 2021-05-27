#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/kernels.hpp>

#ifdef __CUDACC__
  cublasHandle_t  handle;
#endif


int main(){

  array<Index,2> N_xx = {64,64}; // Sizes in space
  array<Index,2> N_vv = {256,256}; // Sizes in velocity

  int r = 5; // rank desired

  double tstar = 20; // final time
  double tau = 0.00625; // time step splitting

  array<double,4> lim_xx = {0.0,4.0*M_PI,0.0,4.0*M_PI}; // Limits for box [ax,bx] x [ay,by] {ax,bx,ay,by}
  array<double,4> lim_vv = {-6.0,6.0,-6.0,6.0}; // Limits for box [av,bv] x [aw,bw] {av,bv,aw,bw}

  double alpha = 0.01;
  double kappa1 = 0.5;
  double kappa2 = 0.5;

  Index nsteps_split = 1; // Number of time steps internal splitting
  Index nsteps_ee = 1; // Number of time steps of exponential euler in internal splitting

  Index nsteps = tstar/tau;
  //nsteps = 100;


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
  ef += 1.0;
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
  efhatx((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
  efhaty((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

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

  coeff_one(lr_sol.V,we_v2,int_v);
  coeff_one(lr_sol.V,we_w2,int_v2);

  int_v += int_v2;

  matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    energy0 += 0.5*(int_x(ii)*rho(ii));
  }

  cout.precision(15);
  cout << std::scientific;

  //ofstream el_energyf;
  //ofstream err_massf;
  //ofstream err_energyf;

  //el_energyf.open("el_energy_2d.txt");
  //err_massf.open("err_mass_2d.txt");
  //err_energyf.open("err_energy_2d.txt");

  //el_energyf.precision(16);
  //err_massf.precision(16);
  //err_energyf.precision(16);

  //el_energyf << tstar << endl;
  //el_energyf << tau << endl;

  //// FOR GPU ////

  #ifdef __CUDACC__
  cublasCreate (&handle);

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

  array<cufftHandle,2> plans_d_e = create_plans_2d(N_xx);

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
  multi_array<double,2> D1y_gpu({r,r});

  multi_array<double,1> d_dcv_r({r},stloc::device);
  multi_array<double,2> d_Tv({r,r},stloc::device);
  multi_array<double,1> d_dcw_r({r},stloc::device);
  multi_array<double,2> d_Tw({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Mhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Nhat({dvvh_mult,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Tvc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Twc({r,r},stloc::device);

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
  multi_array<double,2> d_tmpV({dvv_mult,r}, stloc::device);

  multi_array<cuDoubleComplex,2> d_tmpXhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_tmpVhat({dvvh_mult,r},stloc::device);

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

  //ofstream el_energyGPUf;
  //ofstream err_massGPUf;
  //ofstream err_energyGPUf;

  //el_energyGPUf.open("el_energy_gpu_2d.txt");
  //err_massGPUf.open("err_mass_gpu_2d.txt");
  //err_energyGPUf.open("err_energy_gpu_2d.txt");

  //el_energyGPUf.precision(16);
  //err_massGPUf.precision(16);
  //err_energyGPUf.precision(16);

  //el_energyGPUf << tstar << endl;
  //el_energyGPUf << tau << endl;

  #endif

  // Main cycle
  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i + 1 << " on " << nsteps << endl;

    // CPU

    tmpX = lr_sol.X;

    matmul(tmpX,lr_sol.S,lr_sol.X);

    // Electric field

    coeff_one(lr_sol.V,-h_vv[0]*h_vv[1],rho);

    matvec(lr_sol.X,rho,ef);

    ef += 1.0;

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
    efhatx((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);
    efhaty((N_xx[1]/2)*(N_xx[0]/2+1)) = complex<double>(0.0,0.0);

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

        //Khat += tmpXhat;
        Khat = Khat + tmpXhat; // Operator += to be adjusted

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

      matmul(Lhat,Twc,Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.V,v.begin(),Lv);

        ptw_mult_row(lr_sol.V,w.begin(),Lw);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());

        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());

        matmul_transb(Lvhat,C2vc,Lhat);

        matmul_transb(Lwhat,C2wc,tmpVhat);

        //Lhat += tmpVhat;
        Lhat = Lhat + tmpVhat; // Operator += to be adjusted

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

    //err_massf << err_mass << endl;

    // Error energy

    coeff_one(lr_sol.V,we_v2,int_v);

    coeff_one(lr_sol.V,we_w2,int_v2);

    int_v += int_v2;

    matvec(lr_sol.S,int_v,rho);

    energy = el_energy;
    for(int ii = 0; ii < r; ii++){
      energy += 0.5*(int_x(ii)*rho(ii));
    }

    err_energy = abs(energy0-energy);

  //  err_energyf << err_energy << endl;

    #ifdef __CUDACC__
    d_tmpX = d_lr_sol.X;

    matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);

    coeff_one(d_lr_sol.V,-h_vv[0]*h_vv[1],d_rho);

    matvec(d_lr_sol.X,d_rho,d_ef);

    d_ef += 1.0;

    cufftExecD2Z(plans_d_e[0],d_ef.begin(),(cufftDoubleComplex*)d_efhat.begin());

    der_fourier_2d<<<(dxxh_mult+n_threads-1)/n_threads,n_threads>>>(dxxh_mult, N_xx[0]/2+1, N_xx[1], d_efhat.begin(), d_lim_xx, ncxx, d_efhatx.begin(),d_efhaty.begin());

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatx.begin(),d_efx.begin());

    cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhaty.begin(),d_efy.begin());

    coeff(d_lr_sol.V, d_lr_sol.V, d_we_v.begin(), d_C1v);

    coeff(d_lr_sol.V, d_lr_sol.V, d_we_w.begin(), d_C1w);

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

      exact_sol_exp_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(), d_dcv_r.begin(), ts_split, d_lim_xx, ncxx);

      matmul_transb(d_Mhat,d_Tvc,d_Khat);

      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      // Full step -- Exponential Euler
      //cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin()); // check if CUDA preserves input and eventually adapt. YES

      ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);

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

        exp_euler_fourier_2d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], d_Mhat.begin(),d_dcw_r.begin(),ts_ee, d_lim_xx, d_tmpXhat.begin());

        matmul_transb(d_Mhat,d_Twc,d_Khat);

        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

      }
    }

    gram_schmidt_gpu(d_lr_sol.X, d_lr_sol.S, h_xx[0]*h_xx[1]);

    ptw_mult_scal<<<(d_efx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efx.num_elements(), d_efx.begin(), h_xx[0] * h_xx[1], d_we_x.begin());
    ptw_mult_scal<<<(d_efy.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efy.num_elements(), d_efy.begin(), h_xx[0] * h_xx[1], d_we_y.begin());

    coeff(d_lr_sol.X, d_lr_sol.X, d_we_x.begin(), d_D1x);

    coeff(d_lr_sol.X, d_lr_sol.X, d_we_y.begin(), d_D1y);

    cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_tmpXhat.begin()); // check if CUDA preserves input and eventually adapt

    ptw_mult_row_cplx_fourier_2d<<<(dxxh_mult*r+n_threads-1)/n_threads,n_threads>>>(dxxh_mult*r, N_xx[0]/2+1, N_xx[1], d_tmpXhat.begin(), d_lim_xx, ncxx, d_dXhat_x.begin(), d_dXhat_y.begin()); // Very similar, maybe can be put together

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_x.begin(),d_dX_x.begin());

    cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_y.begin(),d_dX_y.begin());

    coeff(d_lr_sol.X, d_dX_x, h_xx[0]*h_xx[1], d_D2x);

    coeff(d_lr_sol.X, d_dX_y, h_xx[0]*h_xx[1], d_D2y);

    // Explicit Euler
    for(Index jj = 0; jj< nsteps_split; jj++){
      matmul_transb(d_lr_sol.S,d_C1v,d_tmpS);

      matmul(d_D2x,d_tmpS,d_Tv);

      matmul_transb(d_lr_sol.S,d_C1w,d_tmpS);

      matmul(d_D2y,d_tmpS,d_Tw);

      ptw_sum<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(),d_Tv.begin(),d_Tw.begin());

      matmul_transb(d_lr_sol.S,d_C2v,d_tmpS);

      matmul(d_D1x,d_tmpS,d_Tw);

      ptw_diff<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(),d_Tv.begin(),d_Tw.begin());

      matmul_transb(d_lr_sol.S,d_C2w,d_tmpS);

      matmul(d_D1y,d_tmpS,d_Tw);

      expl_euler<<<(d_lr_sol.S.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.S.num_elements(), d_lr_sol.S.begin(), ts_split, d_Tv.begin(), d_Tw.begin());

    }

    d_tmpV = d_lr_sol.V;

    matmul_transb(d_tmpV,d_lr_sol.S,d_lr_sol.V);

    D1x_gpu = d_D1x;
    schur(D1x_gpu, Tv_gpu, dcv_r_gpu, lwork);
    d_Tv = Tv_gpu;
    d_dcv_r = dcv_r_gpu;

    D1y_gpu = d_D1y;
    schur(D1y_gpu, Tw_gpu, dcw_r_gpu, lwork);
    d_Tw = Tw_gpu;
    d_dcw_r = dcw_r_gpu;

    cplx_conv<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(), d_Tv.begin(), d_Tvc.begin());

    cplx_conv<<<(d_Tw.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tw.num_elements(), d_Tw.begin(), d_Twc.begin());

    cplx_conv<<<(d_D2x.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2x.num_elements(), d_D2x.begin(), d_C2vc.begin());

    cplx_conv<<<(d_D2y.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2y.num_elements(), d_D2y.begin(), d_C2wc.begin());

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution
      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

      matmul(d_Lhat,d_Tvc,d_Nhat);

      exact_sol_exp_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(), d_dcv_r.begin(), -ts_split, d_lim_vv, ncvv);

      matmul_transb(d_Nhat,d_Tvc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      // Full step -- Exponential euler
      //cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin()); // check if CUDA preserves input and eventually adapt
      ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), 1.0/ncvv);


      matmul(d_Lhat,d_Twc,d_Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());

        ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());

        cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin()); // check if CUDA preserves input and eventually adapt

        cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin()); // check if CUDA preserves input and eventually adapt

        matmul_transb(d_Lvhat,d_C2vc,d_Lhat);

        matmul_transb(d_Lwhat,d_C2wc,d_tmpVhat);

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        matmul(d_Lhat,d_Twc,d_tmpVhat);

        exp_euler_fourier_2d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], d_Nhat.begin(),d_dcw_r.begin(),-ts_ee, d_lim_vv, d_tmpVhat.begin());

        matmul_transb(d_Nhat,d_Twc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      }
    }

    gram_schmidt_gpu(d_lr_sol.V, d_lr_sol.S, h_vv[0]*h_vv[1]);

    transpose_inplace<<<d_lr_sol.S.num_elements(),1>>>(r,d_lr_sol.S.begin());

    // Electric energy

    cublasDdot (handle, d_efx.num_elements(), d_efx.begin(), 1, d_efx.begin(), 1, d_el_energy_x);
    cublasDdot (handle, d_efy.num_elements(), d_efy.begin(), 1, d_efy.begin(), 1, d_el_energy_y);

    ptw_sum<<<1,1>>>(1,d_el_energy_x,d_el_energy_y); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
    scale_unique<<<1,1>>>(d_el_energy_x,0.5*h_xx[0]*h_xx[1]); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_el_energy_CPU,d_el_energy_x,sizeof(double),cudaMemcpyDeviceToHost);

    //el_energyGPUf << d_el_energy_CPU << endl;

    // Error in mass

    coeff_one(d_lr_sol.X,h_xx[0]*h_xx[1],d_int_x);

    coeff_one(d_lr_sol.V,h_vv[0]*h_vv[1],d_int_v);

    matvec(d_lr_sol.S,d_int_v,d_rho);

    cublasDdot (handle, r, d_int_x.begin(), 1, d_rho.begin(), 1,d_mass);

    cudaMemcpy(&d_mass_CPU,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    err_mass_CPU = abs(mass0-d_mass_CPU);
    //err_massGPUf << err_mass_CPU << endl;

    // Error in energy
    coeff_one(d_lr_sol.V,d_we_v2,d_int_v);

    coeff_one(d_lr_sol.V,d_we_w2,d_int_v2);

    ptw_sum<<<(d_int_v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_int_v.num_elements(),d_int_v.begin(),d_int_v2.begin());


    matvec(d_lr_sol.S,d_int_v,d_rho);


    cublasDdot (handle, r, d_int_x.begin(), 1, d_rho.begin(), 1, d_energy);
    scale_unique<<<1,1>>>(d_energy,0.5); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_energy_CPU,d_energy,sizeof(double),cudaMemcpyDeviceToHost);

    err_energy_CPU = abs(energy0-(d_energy_CPU+d_el_energy_CPU));
    //err_energyGPUf << err_energy_CPU << endl;

    #endif

    cout << "Electric energy: " << el_energy << endl;
    #ifdef __CUDACC__
    cout << "Electric energy GPU: " << d_el_energy_CPU << endl;
    #endif

    cout << "Error in mass: " << err_mass << endl;
    #ifdef __CUDACC__
    cout << "Error in mass GPU: " << err_mass_CPU << endl;
    #endif

    cout << "Error in energy: " << err_energy << endl;
    #ifdef __CUDACC__
    cout << "Error in energy GPU: " << err_energy_CPU << endl;
    #endif

  }

  destroy_plans(plans_e);
  destroy_plans(plans_xx);
  destroy_plans(plans_vv);

  //el_energyf.close();
  //err_massf.close();
  //err_energyf.close();

  #ifdef __CUDACC__
  cublasDestroy(handle);

  destroy_plans(plans_d_e);
  destroy_plans(d_plans_xx);
  destroy_plans(d_plans_vv);

  //el_energyGPUf.close();
  //err_massGPUf.close();
  //err_energyGPUf.close();
  #endif

  return 0;
}
