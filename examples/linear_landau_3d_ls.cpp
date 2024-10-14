#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/kernels.hpp>
#include <generic/timer.hpp>
#include <generic/fft.hpp>

using namespace Ensign;

double tot_gpu_mem = 0.0;


int main(){
  gt::start("Initialization CPU");

  array<Index,3> N_xx = {64,64,64}; // Sizes in space
  array<Index,3> N_vv = {256,256,256}; // Sizes in velocity
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
  nsteps = 1;

  double ts_split = tau / nsteps_split;
  double ts_ee = ts_split / nsteps_ee;

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

  blas_ops blas;
  initialize(lr_sol, X, V, ip_xx, ip_vv, blas); // Mandatory to initialize after plan creation as we use FFTW_MEASURE

  multi_array<double,2> Xsolinit(lr_sol.X);
  multi_array<double,2> Ssolinit(lr_sol.S);
  multi_array<double,2> Vsolinit(lr_sol.V);

  gt::stop("Initialization CPU");
  cout.precision(15);
  cout << std::scientific;

  gt::start("Initialization QOI CPU");
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
  blas.matvec(lr_sol.X,rho,ef);
  ef += 1.0;
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

  integrate(lr_sol.V,we_v2,int_v,blas);
  integrate(lr_sol.V,we_w2,int_v2,blas);

  int_v += int_v2;

  integrate(lr_sol.V,we_u2,int_v2,blas);

  int_v += int_v2;

  blas.matvec(lr_sol.S,int_v,rho);

  for(int ii = 0; ii < r; ii++){
    energy0 += 0.5*(int_x(ii)*rho(ii));
  }
  gt::stop("Initialization QOI CPU");

  //ofstream el_energyf;
  //ofstream err_massf;
  //ofstream err_energyf;

  //el_energyf.open("el_energy_3d.txt");
  //err_massf.open("err_mass_3d.txt");
  //err_energyf.open("err_energy_3d.txt");

  //el_energyf.precision(16);
  //err_massf.precision(16);
  //err_energyf.precision(16);

  //el_energyf << tstar << endl;
  //el_energyf << tau << endl;

  #ifdef __CUDACC__

  array<cufftHandle,2> plans_d_e = create_plans_3d(N_xx,1);
  array<cufftHandle,2> d_plans_xx = create_plans_3d(N_xx, r);
  array<cufftHandle,2> d_plans_vv = create_plans_3d(N_vv, r);

  gt::start("Initialization GPU");

  // Stuff taken from CPU
  lr2<double> d_lr_sol(r,{dxx_mult,dvv_mult},stloc::device);
  d_lr_sol.X = lr_sol.X;
  d_lr_sol.V = lr_sol.V;
  d_lr_sol.S = lr_sol.S;

  multi_array<double,1> d_we_v({dvv_mult},stloc::device);
  d_we_v = we_v;
  multi_array<double,1> d_we_w({dvv_mult},stloc::device);
  d_we_w = we_w;
  multi_array<double,1> d_we_u({dvv_mult},stloc::device);
  d_we_u = we_u;

  multi_array<double,1> d_v({dvv_mult},stloc::device);
  d_v = v;
  multi_array<double,1> d_w({dvv_mult},stloc::device);
  d_w = w;
  multi_array<double,1> d_u({dvv_mult},stloc::device);
  d_u = u;

  multi_array<double,1> d_we_v2({dvv_mult},stloc::device);
  multi_array<double,1> d_we_w2({dvv_mult},stloc::device);
  multi_array<double,1> d_we_u2({dvv_mult},stloc::device);
  d_we_v2 = we_v2;
  d_we_w2 = we_w2;
  d_we_u2 = we_u2;

  // Large stuff
  multi_array<double,1> d_efx({dxx_mult},stloc::device);
  multi_array<double,1> d_efy({dxx_mult},stloc::device);
  multi_array<double,1> d_efz({dxx_mult},stloc::device);

  multi_array<cuDoubleComplex,2> d_tmpXhat({dxxh_mult,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_tmpVhat({dvvh_mult,r},stloc::device);

  // Small stuff
  multi_array<double,1> d_rho({r},stloc::device);

  double* d_lim_xx;
  cudaMalloc((void**)&d_lim_xx,6*sizeof(double));
  cudaMemcpy(d_lim_xx,lim_xx.begin(),6*sizeof(double),cudaMemcpyHostToDevice);

  double* d_lim_vv;
  cudaMalloc((void**)&d_lim_vv,6*sizeof(double));
  cudaMemcpy(d_lim_vv,lim_vv.begin(),6*sizeof(double),cudaMemcpyHostToDevice);

  multi_array<double,2> d_C1v({r,r},stloc::device);
  multi_array<double,2> d_C1w({r,r}, stloc::device);
  multi_array<double,2> d_C1u({r,r}, stloc::device);

  multi_array<double,2> d_C2v({r,r},stloc::device);
  multi_array<double,2> d_C2w({r,r},stloc::device);
  multi_array<double,2> d_C2u({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_C2vc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2wc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_C2uc({r,r},stloc::device);

  multi_array<double,2> d_D1x({r,r}, stloc::device);
  multi_array<double,2> d_D1y({r,r}, stloc::device);
  multi_array<double,2> d_D1z({r,r}, stloc::device);

  multi_array<double,2> d_D2x({r,r},stloc::device);
  multi_array<double,2> d_D2y({r,r},stloc::device);
  multi_array<double,2> d_D2z({r,r},stloc::device);

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
  multi_array<double,2> D1y_gpu({r,r});
  multi_array<double,2> D1z_gpu({r,r});

  multi_array<double,1> d_dcv_r({r},stloc::device);
  multi_array<double,2> d_Tv({r,r},stloc::device);
  multi_array<double,1> d_dcw_r({r},stloc::device);
  multi_array<double,2> d_Tw({r,r},stloc::device);
  multi_array<double,1> d_dcu_r({r},stloc::device);
  multi_array<double,2> d_Tu({r,r},stloc::device);

  multi_array<cuDoubleComplex,2> d_Tvc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Twc({r,r},stloc::device);
  multi_array<cuDoubleComplex,2> d_Tuc({r,r},stloc::device);

  multi_array<double,2> d_tmpS({r,r}, stloc::device);

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

  multi_array<double,1> d_int_v2({r},stloc::device);
  multi_array<double,1> d_int_v3({r},stloc::device);

  double* d_energy;
  cudaMalloc((void**)&d_energy,sizeof(double));
  double d_energy_CPU;
  double err_energy_CPU;

  cudaDeviceSynchronize();
  gt::stop("Initialization GPU");
/*
  cout << "Tot GPU mem alloc: " << tot_gpu_mem << " GB" << endl;

  bool cio = true;
  while (cio == true){

  }

  cout << "INTERRUPTED!" << endl;
  exit(1);
*/



  //ofstream el_energyGPUf;
  //ofstream err_massGPUf;
  //ofstream err_energyGPUf;

  //el_energyGPUf.open("el_energy_gpu_3d.txt");
  //err_massGPUf.open("err_mass_gpu_3d.txt");
  //err_energyGPUf.open("err_energy_gpu_3d.txt");

  //el_energyGPUf.precision(16);
  //err_massGPUf.precision(16);
  //err_energyGPUf.precision(16);

  //el_energyGPUf << tstar << endl;
  //el_energyGPUf << tau << endl;

  #endif

  for(Index i = 0; i < nsteps; i++){

    cout << "Time step " << i + 1 << " on " << nsteps << endl;

    gt::start("Main loop CPU");

    tmpX = lr_sol.X;

    blas.matmul(tmpX,lr_sol.S,lr_sol.X);

    gt::start("Electric Field CPU");
    // Electric field

    //gt::start("Electric Field CPU - coeff");
    integrate(lr_sol.V,-h_vv[0]*h_vv[1]*h_vv[2],rho,blas);

    //gt::stop("Electric Field CPU - coeff");

    //gt::start("Electric Field CPU - matvec");
    blas.matvec(lr_sol.X,rho,ef);

    //gt::stop("Electric Field CPU - matvec");
    ef += 1.0;

    fftw_execute_dft_r2c(plans_e[0],ef.begin(),(fftw_complex*)efhat.begin());

    gt::start("Electric Field CPU - ptw");
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

    gt::stop("Electric Field CPU - ptw");

    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatx.begin(),efx.begin());
    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhaty.begin(),efy.begin());
    fftw_execute_dft_c2r(plans_e[1],(fftw_complex*)efhatz.begin(),efz.begin());

    gt::stop("Electric Field CPU");
    // Main of K step

    gt::start("C coeff CPU");
    coeff(lr_sol.V, lr_sol.V, we_v.begin(), C1v, blas);

    coeff(lr_sol.V, lr_sol.V, we_w.begin(), C1w, blas);

    coeff(lr_sol.V, lr_sol.V, we_u.begin(), C1u, blas);

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

    gt::start("K step CPU");

    gt::start("Schur K CPU");
    schur(C1v, Tv, dcv_r);
    schur(C1w, Tw, dcw_r);
    schur(C1u, Tu, dcu_r);

    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    Tu.to_cplx(Tuc);
    C2v.to_cplx(C2vc);
    C2w.to_cplx(C2wc);
    C2u.to_cplx(C2uc);

    gt::stop("Schur K CPU");

    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution

      gt::start("First split K CPU");
      fftw_execute_dft_r2c(plans_xx[0],lr_sol.X.begin(),(fftw_complex*)Khat.begin());

      blas.matmul(Khat,Tvc,Mhat);

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


      blas.matmul_transb(Mhat,Tvc,Khat);

      gt::stop("First split K CPU");
      // Full step -- Exact solution

      gt::start("Second split K CPU");
      blas.matmul(Khat,Twc,Mhat);

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

      blas.matmul_transb(Mhat,Twc,Khat);

      fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

      gt::stop("Second split K CPU");
      // Full step -- Exponential Euler

      gt::start("Third split K CPU");
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

        //Khat += tmpXhat;
        Khat = Khat + tmpXhat;

        blas.matmul_transb(Kezhat,C2uc,tmpXhat);

        //Khat += tmpXhat;
        Khat = Khat + tmpXhat;

        //gt::start("EE Third split K CPU");
        blas.matmul(Khat,Tuc,tmpXhat);

        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_xx[2]; k++){
            if(k < (N_xx[2]/2)) { mult_k = k; } else if(k == (N_xx[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_xx[2]); }
            for(Index j = 0; j < N_xx[1]; j++){
              for(Index i = 0; i < (N_xx[0]/2 + 1); i++){
                complex<double> lambdaz = complex<double>(0.0,2.0*M_PI/(lim_xx[5]-lim_xx[4])*mult_k);

                Index idx = i+j*(N_xx[0]/2+1) + k*((N_xx[0]/2+1)*N_xx[1]);

                Mhat(idx,rr) *= exp(-ts_ee*lambdaz*dcu_r(rr));
                Mhat(idx,rr) += ts_ee*phi1_im(-ts_ee*lambdaz*dcu_r(rr))*tmpXhat(idx,rr);
              }
            }
          }
        }

        blas.matmul_transb(Mhat,Tuc,Khat);

        Khat *= ncxx;


        fftw_execute_dft_c2r(plans_xx[1],(fftw_complex*)Khat.begin(),lr_sol.X.begin());

        //gt::stop("EE Third split K CPU");
      }

      gt::stop("Third split K CPU");
    }

    gt::start("Gram Schmidt K CPU");
    orthogonalize(lr_sol.X, lr_sol.S, ip_xx);
    gt::stop("Gram Schmidt K CPU");

    gt::stop("K step CPU");
    // S Step

    gt::start("D coeff CPU");
    for(Index j = 0; j < (dxx_mult); j++){
      we_x(j) = efx(j) * h_xx[0] * h_xx[1] * h_xx[2];
      we_y(j) = efy(j) * h_xx[0] * h_xx[1] * h_xx[2];
      we_z(j) = efz(j) * h_xx[0] * h_xx[1] * h_xx[2];
    }


    coeff(lr_sol.X, lr_sol.X, we_x.begin(), D1x, blas);
    coeff(lr_sol.X, lr_sol.X, we_y.begin(), D1y, blas);
    coeff(lr_sol.X, lr_sol.X, we_z.begin(), D1z, blas);


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
    // Explicit Euler
    for(Index jj = 0; jj< nsteps_split; jj++){
      blas.matmul_transb(lr_sol.S,C1v,tmpS);

      blas.matmul(D2x,tmpS,Tv);

      blas.matmul_transb(lr_sol.S,C1w,tmpS);

      blas.matmul(D2y,tmpS,Tw);

      blas.matmul_transb(lr_sol.S,C1u,tmpS);

      blas.matmul(D2z,tmpS,Tu);

      Tv += Tw;
      Tv += Tu;


      blas.matmul_transb(lr_sol.S,C2v,tmpS);

      blas.matmul(D1x,tmpS,Tw);

      Tv -= Tw;


      blas.matmul_transb(lr_sol.S,C2w,tmpS);

      blas.matmul(D1y,tmpS,Tw);

      Tv -= Tw;

      blas.matmul_transb(lr_sol.S,C2u,tmpS);

      blas.matmul(D1z,tmpS,Tw);

      Tv -= Tw;

      Tv *= ts_split;
      lr_sol.S += Tv;

    }

    gt::stop("S step CPU");
    // L step - here we reuse some old variable names

    tmpV = lr_sol.V;

    blas.matmul_transb(tmpV,lr_sol.S,lr_sol.V);

    gt::start("L step CPU");

    gt::start("Schur L CPU");
    schur(D1x, Tv, dcv_r);
    schur(D1y, Tw, dcw_r);
    schur(D1z, Tu, dcu_r);


    Tv.to_cplx(Tvc);
    Tw.to_cplx(Twc);
    Tu.to_cplx(Tuc);
    D2x.to_cplx(C2vc);
    D2y.to_cplx(C2wc);
    D2z.to_cplx(C2uc);

    gt::stop("Schur L CPU");
    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){

      gt::start("First split L CPU");
      // Full step -- Exact solution
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

      blas.matmul(Lhat,Tvc,Nhat);

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


      blas.matmul_transb(Nhat,Tvc,Lhat);

      gt::stop("First split L CPU");

      gt::start("Second split L CPU");
      blas.matmul(Lhat,Twc,Nhat);

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

      blas.matmul_transb(Nhat,Twc,Lhat);

      fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      gt::stop("Second split L CPU");

      gt::start("Third split L CPU");
      // Full step -- Exponential euler
      fftw_execute_dft_r2c(plans_vv[0],lr_sol.V.begin(),(fftw_complex*)Lhat.begin());

      blas.matmul(Lhat,Tuc,Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        ptw_mult_row(lr_sol.V,v,Lv);
        ptw_mult_row(lr_sol.V,w,Lw);
        ptw_mult_row(lr_sol.V,u,Lu);

        fftw_execute_dft_r2c(plans_vv[0],Lv.begin(),(fftw_complex*)Lvhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lw.begin(),(fftw_complex*)Lwhat.begin());
        fftw_execute_dft_r2c(plans_vv[0],Lu.begin(),(fftw_complex*)Luhat.begin());

        blas.matmul_transb(Lvhat,C2vc,Lhat);

        blas.matmul_transb(Lwhat,C2wc,tmpVhat);

        //Lhat += tmpVhat;
        Lhat = Lhat + tmpVhat;

        blas.matmul_transb(Luhat,C2uc,tmpVhat);

        //Lhat += tmpVhat;
        Lhat = Lhat + tmpVhat;

        blas.matmul(Lhat,Tuc,tmpVhat);

        for(int rr = 0; rr < r; rr++){
          for(Index k = 0; k < N_vv[2]; k++){
            if(k < (N_vv[2]/2)) { mult_k = k; } else if(k == (N_vv[2]/2)) { mult_k = 0.0; } else { mult_k = (k-N_vv[2]); }
            for(Index j = 0; j < N_vv[1]; j++){
              for(Index i = 0; i < (N_vv[0]/2 + 1); i++){
                complex<double> lambdau = complex<double>(0.0,2.0*M_PI/(lim_vv[5]-lim_vv[4])*mult_k);
                Index idx = i+j*(N_vv[0]/2+1) + k*((N_vv[0]/2+1)*N_vv[1]);

                Nhat(idx,rr) *= exp(ts_ee*lambdau*dcu_r(rr));
                Nhat(idx,rr) -= ts_ee*phi1_im(ts_ee*lambdau*dcu_r(rr))*tmpVhat(idx,rr);
              }
            }
          }
        }

        blas.matmul_transb(Nhat,Tuc,Lhat);

        Lhat *= ncvv;

        fftw_execute_dft_c2r(plans_vv[1],(fftw_complex*)Lhat.begin(),lr_sol.V.begin());

      }

      gt::stop("Third split L CPU");

    }

    gt::start("Gram Schmidt L CPU");
    orthogonalize(lr_sol.V, lr_sol.S, ip_vv);

    gt::stop("Gram Schmidt L CPU");


    gt::start("Transpose S CPU");
    transpose_inplace(lr_sol.S);
    gt::stop("Transpose S CPU");

    gt::stop("L step CPU");

    gt::stop("Main loop CPU");

    gt::start("Quantities CPU");
    // Electric energy
    el_energy = 0.0;
    for(Index ii = 0; ii < (dxx_mult); ii++){
      el_energy += 0.5*(pow(efx(ii),2)+pow(efy(ii),2)+pow(efz(ii),2))*h_xx[0]*h_xx[1]*h_xx[2];
    }

    //el_energyf << el_energy << endl;

    // Error Mass
    integrate(lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],int_x,blas);
    integrate(lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],int_v,blas);

    blas.matvec(lr_sol.S,int_v,rho);

    mass = 0.0;
    for(int ii = 0; ii < r; ii++){
      mass += (int_x(ii)*rho(ii));
    }

    err_mass = abs(mass0-mass);

  //  err_massf << err_mass << endl;

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

  //  err_energyf << err_energy << endl;

    gt::stop("Quantities CPU");
    #ifdef __CUDACC__

    // K step

    gt::start("Main loop GPU");
    {
      multi_array<double,2> d_tmpX({dxx_mult,r},stloc::device);
      d_tmpX = d_lr_sol.X;
      blas.matmul(d_tmpX,d_lr_sol.S,d_lr_sol.X);
    }

    gt::start("Electric Field GPU");

//    gt::start("Electric Field GPU - coeff");
    integrate(d_lr_sol.V,-h_vv[0]*h_vv[1]*h_vv[2],d_rho,blas);
  //  cudaDeviceSynchronize();
  //  gt::stop("Electric Field GPU - coeff");

    //gt::start("Electric Field GPU - matvec");
    blas.matvec(d_lr_sol.X,d_rho,d_efx);
    //cudaDeviceSynchronize();
    //gt::stop("Electric Field GPU - matvec");

    //gt::start("Electric Field GPU - FFT");
    d_efx += 1.0;
    {
      multi_array<cuDoubleComplex,1> d_efhat({dxxh_mult},stloc::device);
      cufftExecD2Z(plans_d_e[0],d_efx.begin(),(cufftDoubleComplex*)d_efhat.begin());

      multi_array<cuDoubleComplex,1> d_efhatx({dxxh_mult},stloc::device);
      {
        multi_array<cuDoubleComplex,1> d_efhaty({dxxh_mult},stloc::device);
        {
          multi_array<cuDoubleComplex,1> d_efhatz({dxxh_mult},stloc::device);

          der_fourier_3d<<<(dxxh_mult+n_threads-1)/n_threads,n_threads>>>(dxxh_mult, N_xx[0]/2+1, N_xx[1], N_xx[2], d_efhat.begin(), d_lim_xx, ncxx, d_efhatx.begin(), d_efhaty.begin(), d_efhatz.begin());

          cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatz.begin(),d_efz.begin());
        }
        cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhaty.begin(),d_efy.begin());
      }
      cufftExecZ2D(plans_d_e[1],(cufftDoubleComplex*)d_efhatx.begin(),d_efx.begin());
    }

    //cudaDeviceSynchronize();
    //gt::stop("Electric Field GPU - FFT");
    cudaDeviceSynchronize();
    gt::stop("Electric Field GPU");

    gt::start("C coeff GPU");
    coeff(d_lr_sol.V, d_lr_sol.V, d_we_v.begin(), d_C1v, blas);

    coeff(d_lr_sol.V, d_lr_sol.V, d_we_w.begin(), d_C1w, blas);

    coeff(d_lr_sol.V, d_lr_sol.V, d_we_u.begin(), d_C1u, blas);

    cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_tmpVhat.begin());

  {
    multi_array<cuDoubleComplex,2> d_dVhat_v({dvvh_mult,r},stloc::device);
    multi_array<double,2> d_dV_v({dvv_mult,r},stloc::device);
    {
      multi_array<cuDoubleComplex,2> d_dVhat_w({dvvh_mult,r},stloc::device);
      multi_array<double,2> d_dV_w({dvv_mult,r},stloc::device);

      {
        multi_array<cuDoubleComplex,2> d_dVhat_u({dvvh_mult,r},stloc::device);
        multi_array<double,2> d_dV_u({dvv_mult,r},stloc::device);
        ptw_mult_row_cplx_fourier_3d<<<(dvvh_mult*r+n_threads-1)/n_threads,n_threads>>>(dvvh_mult*r, N_vv[0]/2+1, N_vv[1], N_vv[2], d_tmpVhat.begin(), d_lim_vv, ncvv, d_dVhat_v.begin(), d_dVhat_w.begin(), d_dVhat_u.begin());

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_u.begin(),d_dV_u.begin());
        coeff(d_lr_sol.V, d_dV_u, h_vv[0]*h_vv[1]*h_vv[2], d_C2u, blas);

      }
      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_w.begin(),d_dV_w.begin());
      coeff(d_lr_sol.V, d_dV_w, h_vv[0]*h_vv[1]*h_vv[2], d_C2w, blas);

    }
    cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_dVhat_v.begin(),d_dV_v.begin());
    coeff(d_lr_sol.V, d_dV_v, h_vv[0]*h_vv[1]*h_vv[2], d_C2v, blas);
  }

    cudaDeviceSynchronize();
    gt::stop("C coeff GPU");

    gt::start("K step GPU");


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
    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){
      // Full step -- Exact solution

      gt::start("First split K GPU");
      {
        multi_array<cuDoubleComplex,2> d_Khat({dxxh_mult,r},stloc::device);
        multi_array<cuDoubleComplex,2> d_Mhat({dxxh_mult,r},stloc::device);

        cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());

        blas.matmul(d_Khat,d_Tvc,d_Mhat);

        exact_sol_exp_3d_a<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcv_r.begin(), ts_split, d_lim_xx);

        blas.matmul_transb(d_Mhat,d_Tvc,d_Khat);

        cudaDeviceSynchronize();
        gt::stop("First split K GPU");
        // Full step -- Exact solution

        gt::start("Second split K GPU");
        blas.matmul(d_Khat,d_Twc,d_Mhat);

        exact_sol_exp_3d_b<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(), d_dcw_r.begin(), ts_split, d_lim_xx, ncxx);

        blas.matmul_transb(d_Mhat,d_Twc,d_Khat);

        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin()); // in gpu remember to divide so we avoid the last fft
        // actually it depends on the cuda version, in some the input is preserved, in others it is not. So do fft in general
        cudaDeviceSynchronize();
        gt::stop("Second split K GPU");
        // Full step -- Exponential Euler

        gt::start("Third split K GPU");
        //cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_Khat.begin());
        ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), 1.0/ncxx);


        blas.matmul(d_Khat,d_Tuc,d_Mhat);

        for(Index jj = 0; jj < nsteps_ee; jj++){
          {
            multi_array<double,2> d_Kex({dxx_mult,r},stloc::device);
            multi_array<cuDoubleComplex,2> d_Kexhat({dxxh_mult,r},stloc::device);

            ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efx.begin(),d_Kex.begin());
            cufftExecD2Z(d_plans_xx[0],d_Kex.begin(),(cufftDoubleComplex*)d_Kexhat.begin());
            blas.matmul_transb(d_Kexhat,d_C2vc,d_Khat);
          }

          {
            multi_array<double,2> d_Key({dxx_mult,r},stloc::device);
            multi_array<cuDoubleComplex,2> d_Keyhat({dxxh_mult,r},stloc::device);

            ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efy.begin(),d_Key.begin());
            cufftExecD2Z(d_plans_xx[0],d_Key.begin(),(cufftDoubleComplex*)d_Keyhat.begin());
            blas.matmul_transb(d_Keyhat,d_C2wc,d_tmpXhat);
            ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

          }

          {
            multi_array<double,2> d_Kez({dxx_mult,r},stloc::device);
            multi_array<cuDoubleComplex,2> d_Kezhat({dxxh_mult,r},stloc::device);

            ptw_mult_row_k<<<(d_lr_sol.X.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.X.num_elements(),d_lr_sol.X.shape()[0],d_lr_sol.X.begin(),d_efz.begin(),d_Kez.begin());
            cufftExecD2Z(d_plans_xx[0],d_Kez.begin(),(cufftDoubleComplex*)d_Kezhat.begin());
            blas.matmul_transb(d_Kezhat,d_C2uc,d_tmpXhat);
          }

          ptw_sum_complex<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), d_tmpXhat.begin());

          blas.matmul(d_Khat,d_Tuc,d_tmpXhat);

          exp_euler_fourier_3d<<<(d_Mhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Mhat.num_elements(), N_xx[0]/2 + 1, N_xx[1], N_xx[2], d_Mhat.begin(),d_dcu_r.begin(),ts_ee, d_lim_xx, d_tmpXhat.begin());

          blas.matmul_transb(d_Mhat,d_Tuc,d_Khat);

          ptw_mult_cplx<<<(d_Khat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Khat.num_elements(), d_Khat.begin(), ncxx);

          cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_Khat.begin(),d_lr_sol.X.begin());

        }
        cudaDeviceSynchronize();
        gt::stop("Third split K GPU");
      }
    }

    gt::start("Gram Schmidt K GPU");
    gram_schmidt_gpu(d_lr_sol.X, d_lr_sol.S, h_xx[0]*h_xx[1]*h_xx[2]);
    cudaDeviceSynchronize();
    gt::stop("Gram Schmidt K GPU");

    gt::stop("K step GPU");
    // S step

    gt::start("D coeff GPU");
  {
    multi_array<double,1> d_we_x({dxx_mult},stloc::device);
    ptw_mult_scal<<<(d_efx.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efx.num_elements(), d_efx.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_x.begin());
    coeff(d_lr_sol.X, d_lr_sol.X, d_we_x.begin(), d_D1x, blas);
  }
  {
    multi_array<double,1> d_we_y({dxx_mult},stloc::device);
    ptw_mult_scal<<<(d_efy.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efy.num_elements(), d_efy.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_y.begin());
    coeff(d_lr_sol.X, d_lr_sol.X, d_we_y.begin(), d_D1y, blas);
  }
  {
    multi_array<double,1> d_we_z({dxx_mult},stloc::device);
    ptw_mult_scal<<<(d_efz.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_efz.num_elements(), d_efz.begin(), h_xx[0] * h_xx[1] * h_xx[2], d_we_z.begin());
    coeff(d_lr_sol.X, d_lr_sol.X, d_we_z.begin(), d_D1z, blas);
  }

    cufftExecD2Z(d_plans_xx[0],d_lr_sol.X.begin(),(cufftDoubleComplex*)d_tmpXhat.begin());

    {
      multi_array<cuDoubleComplex,2> d_dXhat_x({dxxh_mult,r},stloc::device);
      multi_array<double,2> d_dX_x({dxx_mult,r},stloc::device);
      {
        multi_array<cuDoubleComplex,2> d_dXhat_y({dxxh_mult,r},stloc::device);
        multi_array<double,2> d_dX_y({dxx_mult,r},stloc::device);

        {
          multi_array<cuDoubleComplex,2> d_dXhat_z({dxxh_mult,r},stloc::device);
          multi_array<double,2> d_dX_z({dxx_mult,r},stloc::device);
          ptw_mult_row_cplx_fourier_3d<<<(dxxh_mult*r+n_threads-1)/n_threads,n_threads>>>(dxxh_mult*r, N_xx[0]/2+1, N_xx[1], N_xx[2], d_tmpXhat.begin(), d_lim_xx, ncxx, d_dXhat_x.begin(), d_dXhat_y.begin(), d_dXhat_z.begin());

          cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_z.begin(),d_dX_z.begin());
          coeff(d_lr_sol.X, d_dX_z, h_xx[0]*h_xx[1]*h_xx[2], d_D2z, blas);

        }
        cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_y.begin(),d_dX_y.begin());
        coeff(d_lr_sol.X, d_dX_y, h_xx[0]*h_xx[1]*h_xx[2], d_D2y, blas);

      }
      cufftExecZ2D(d_plans_xx[1],(cufftDoubleComplex*)d_dXhat_x.begin(),d_dX_x.begin());
      coeff(d_lr_sol.X, d_dX_x, h_xx[0]*h_xx[1]*h_xx[2], d_D2x, blas);
    }

    cudaDeviceSynchronize();
    gt::stop("D coeff GPU");

    gt::start("S step GPU");
    // Explicit Euler
    for(Index jj = 0; jj< nsteps_split; jj++){
      blas.matmul_transb(d_lr_sol.S,d_C1v,d_tmpS);

      blas.matmul(d_D2x,d_tmpS,d_Tv);

      blas.matmul_transb(d_lr_sol.S,d_C1w,d_tmpS);

      blas.matmul(d_D2y,d_tmpS,d_Tw);

      blas.matmul_transb(d_lr_sol.S,d_C1u,d_tmpS);

      blas.matmul(d_D2z,d_tmpS,d_Tu);

      ptw_sum_3mat<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(),d_Tv.begin(),d_Tw.begin(),d_Tu.begin());

      blas.matmul_transb(d_lr_sol.S,d_C2v,d_tmpS);

      blas.matmul(d_D1x,d_tmpS,d_Tw);

      ptw_diff<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(),d_Tv.begin(),d_Tw.begin());

      blas.matmul_transb(d_lr_sol.S,d_C2w,d_tmpS);

      blas.matmul(d_D1y,d_tmpS,d_Tw);

      ptw_diff<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(),d_Tv.begin(),d_Tw.begin());


      blas.matmul_transb(d_lr_sol.S,d_C2u,d_tmpS);

      blas.matmul(d_D1z,d_tmpS,d_Tw);

      expl_euler<<<(d_lr_sol.S.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.S.num_elements(), d_lr_sol.S.begin(), ts_split, d_Tv.begin(), d_Tw.begin());

    }
    cudaDeviceSynchronize();
    gt::stop("S step GPU");
    // L step

    {
      multi_array<double,2> d_tmpV({dvv_mult,r}, stloc::device);
      d_tmpV = d_lr_sol.V;
      blas.matmul_transb(d_tmpV,d_lr_sol.S,d_lr_sol.V);
    }
    gt::start("L step GPU");

    gt::start("Schur L GPU");

    D1x_gpu = d_D1x;

    schur(D1x_gpu, Tv_gpu, dcv_r_gpu);

    d_Tv = Tv_gpu;
    d_dcv_r = dcv_r_gpu;

    D1y_gpu = d_D1y;

    schur(D1y_gpu, Tw_gpu, dcw_r_gpu);

    d_Tw = Tw_gpu;
    d_dcw_r = dcw_r_gpu;

    D1z_gpu = d_D1z;

    schur(D1z_gpu, Tu_gpu, dcu_r_gpu);

    d_Tu = Tu_gpu;
    d_dcu_r = dcu_r_gpu;

    cplx_conv<<<(d_Tv.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tv.num_elements(), d_Tv.begin(), d_Tvc.begin());
    cplx_conv<<<(d_Tw.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tw.num_elements(), d_Tw.begin(), d_Twc.begin());
    cplx_conv<<<(d_Tu.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Tu.num_elements(), d_Tu.begin(), d_Tuc.begin());

    cplx_conv<<<(d_D2x.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2x.num_elements(), d_D2x.begin(), d_C2vc.begin());
    cplx_conv<<<(d_D2y.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2y.num_elements(), d_D2y.begin(), d_C2wc.begin());
    cplx_conv<<<(d_D2z.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_D2z.num_elements(), d_D2z.begin(), d_C2uc.begin());

    cudaDeviceSynchronize();
    gt::stop("Schur L GPU");
    // Internal splitting
    for(Index ii = 0; ii < nsteps_split; ii++){

      gt::start("First split L GPU");
      // Full step -- Exact solution
      {
        multi_array<cuDoubleComplex,2> d_Lhat({dvvh_mult,r},stloc::device);
        multi_array<cuDoubleComplex,2> d_Nhat({dvvh_mult,r},stloc::device);


      cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin());

      blas.matmul(d_Lhat,d_Tvc,d_Nhat);

      exact_sol_exp_3d_a<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dcv_r.begin(), -ts_split, d_lim_vv);

      blas.matmul_transb(d_Nhat,d_Tvc,d_Lhat);

      cudaDeviceSynchronize();
      gt::stop("First split L GPU");


      gt::start("Second split L GPU");
      blas.matmul(d_Lhat,d_Twc,d_Nhat);

      exact_sol_exp_3d_b<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(), d_dcw_r.begin(), -ts_split, d_lim_vv, ncvv);

      blas.matmul_transb(d_Nhat,d_Twc,d_Lhat);

      cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      cudaDeviceSynchronize();
      gt::stop("Second split L GPU");

      gt::start("Third split L GPU");
      // Full step -- Exponential euler
      //cufftExecD2Z(d_plans_vv[0],d_lr_sol.V.begin(),(cufftDoubleComplex*)d_Lhat.begin()); // REMEMBER TO ADAPT
      ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), 1.0/ncvv);

      blas.matmul(d_Lhat,d_Tuc,d_Nhat);

      for(Index jj = 0; jj < nsteps_ee; jj++){

        {
          multi_array<double,2> d_Lv({dvv_mult,r},stloc::device);
          multi_array<cuDoubleComplex,2> d_Lvhat({dvvh_mult,r},stloc::device);

          ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_v.begin(),d_Lv.begin());
          cufftExecD2Z(d_plans_vv[0],d_Lv.begin(),(cufftDoubleComplex*)d_Lvhat.begin());
          blas.matmul_transb(d_Lvhat,d_C2vc,d_Lhat);
        }

        {
          multi_array<double,2> d_Lw({dvv_mult,r},stloc::device);
          multi_array<cuDoubleComplex,2> d_Lwhat({dvvh_mult,r},stloc::device);

          ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_w.begin(),d_Lw.begin());
          cufftExecD2Z(d_plans_vv[0],d_Lw.begin(),(cufftDoubleComplex*)d_Lwhat.begin());
          blas.matmul_transb(d_Lwhat,d_C2wc,d_tmpVhat);
          ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());
        }

        {
          multi_array<double,2> d_Lu({dvv_mult,r},stloc::device);
          multi_array<cuDoubleComplex,2> d_Luhat({dvvh_mult,r},stloc::device);

          ptw_mult_row_k<<<(d_lr_sol.V.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_lr_sol.V.num_elements(),d_lr_sol.V.shape()[0],d_lr_sol.V.begin(),d_u.begin(),d_Lu.begin());
          cufftExecD2Z(d_plans_vv[0],d_Lu.begin(),(cufftDoubleComplex*)d_Luhat.begin());
          blas.matmul_transb(d_Luhat,d_C2uc,d_tmpVhat);
        }

        ptw_sum_complex<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), d_tmpVhat.begin());

        blas.matmul(d_Lhat,d_Tuc,d_tmpVhat);

        exp_euler_fourier_3d<<<(d_Nhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Nhat.num_elements(), N_vv[0]/2 + 1, N_vv[1], N_vv[2], d_Nhat.begin(),d_dcu_r.begin(),-ts_ee, d_lim_vv, d_tmpVhat.begin());

        blas.matmul_transb(d_Nhat,d_Tuc,d_Lhat);

        ptw_mult_cplx<<<(d_Lhat.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_Lhat.num_elements(), d_Lhat.begin(), ncvv);

        cufftExecZ2D(d_plans_vv[1],(cufftDoubleComplex*)d_Lhat.begin(),d_lr_sol.V.begin());

      }
      cudaDeviceSynchronize();
      gt::stop("Third split L GPU");
      }
    }

    gt::start("Gram Schmidt L GPU");
    gram_schmidt_gpu(d_lr_sol.V, d_lr_sol.S, h_vv[0]*h_vv[1]*h_vv[2]);
    cudaDeviceSynchronize();
    gt::stop("Gram Schmidt L GPU");

    gt::start("Transpose S GPU");
    //transpose_inplace<<<d_lr_sol.S.num_elements(),1>>>(r,d_lr_sol.S.begin());
    transpose_inplace(d_lr_sol.S);
    cudaDeviceSynchronize();

    gt::stop("Transpose S GPU");

    gt::stop("L step GPU");

    gt::stop("Main loop GPU");

    gt::start("Quantities GPU");
   // Electric energy

    //gt::start("Quantities GPU - El en");
    cublasDdot (blas.handle_devres, d_efx.num_elements(), d_efx.begin(), 1, d_efx.begin(), 1, d_el_energy_x);
    cublasDdot (blas.handle_devres, d_efy.num_elements(), d_efy.begin(), 1, d_efy.begin(), 1, d_el_energy_y);
    cublasDdot (blas.handle_devres, d_efz.num_elements(), d_efz.begin(), 1, d_efz.begin(), 1, d_el_energy_z);
    cudaDeviceSynchronize();
    ptw_sum<<<1,1>>>(1,d_el_energy_x,d_el_energy_y);
    ptw_sum<<<1,1>>>(1,d_el_energy_x,d_el_energy_z);

    scale_unique<<<1,1>>>(d_el_energy_x,0.5*h_xx[0]*h_xx[1]*h_xx[2]); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call

    cudaMemcpy(&d_el_energy_CPU,d_el_energy_x,sizeof(double),cudaMemcpyDeviceToHost);

    //el_energyGPUf << d_el_energy_CPU << endl;
    //cudaDeviceSynchronize();
    //gt::stop("Quantities GPU - El en");

    //gt::start("Quantities GPU - mass");
    // Error mass

    //gt::start("Quantities GPU - mass - coeff");
    integrate(d_lr_sol.X,h_xx[0]*h_xx[1]*h_xx[2],d_int_x,blas);
    integrate(d_lr_sol.V,h_vv[0]*h_vv[1]*h_vv[2],d_int_v,blas);

    //cudaDeviceSynchronize();
    //gt::stop("Quantities GPU - mass - coeff");
    blas.matvec(d_lr_sol.S,d_int_v,d_rho);

    cublasDdot (blas.handle_devres, r, d_int_x.begin(), 1, d_rho.begin(), 1,d_mass);
    cudaDeviceSynchronize();

    cudaMemcpy(&d_mass_CPU,d_mass,sizeof(double),cudaMemcpyDeviceToHost);

    err_mass_CPU = abs(mass0-d_mass_CPU);

    //err_massGPUf << err_mass_CPU << endl;
    //cudaDeviceSynchronize();
    //gt::stop("Quantities GPU - mass");

    //gt::start("Quantities GPU - energy");
    // Error energy

    //gt::start("Quantities GPU - energy - coeff");
    integrate(d_lr_sol.V,d_we_v2,d_int_v,blas);
    integrate(d_lr_sol.V,d_we_w2,d_int_v2,blas);
    integrate(d_lr_sol.V,d_we_u2,d_int_v3,blas);

    //cudaDeviceSynchronize();
    //gt::stop("Quantities GPU - energy - coeff");
    ptw_sum_3mat<<<(d_int_v.num_elements()+n_threads-1)/n_threads,n_threads>>>(d_int_v.num_elements(),d_int_v.begin(),d_int_v2.begin(),d_int_v3.begin());

    blas.matvec(d_lr_sol.S,d_int_v,d_rho);

    cublasDdot (blas.handle_devres, r, d_int_x.begin(), 1, d_rho.begin(), 1, d_energy);
    cudaDeviceSynchronize();
    scale_unique<<<1,1>>>(d_energy,0.5); //cudamemcpyDev2Dev seems to be slow, better to use a simple kernel call
    cudaMemcpy(&d_energy_CPU,d_energy,sizeof(double),cudaMemcpyDeviceToHost);

    err_energy_CPU = abs(energy0-(d_energy_CPU+d_el_energy_CPU));

    //err_energyGPUf << err_energy_CPU << endl;
    //cudaDeviceSynchronize();
    //gt::stop("Quantities GPU - energy");
    cudaDeviceSynchronize();
    gt::stop("Quantities GPU");


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
  destroy_plans(plans_d_e);
  destroy_plans(d_plans_xx);
  destroy_plans(d_plans_vv);

  //el_energyGPUf.close();
  //err_massGPUf.close();
  //err_energyGPUf.close();
  #endif

  cout << gt::sorted_output() << endl;

  return 0;
}
