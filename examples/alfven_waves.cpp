#include "alfven_waves.hpp"

#include <cxxopts.hpp>

int main(int argc, char** argv) {

  cxxopts::Options options("alfven_waves", "2+2 dimensional dynamical low-rank solver for Alfven wave problems");
  options.add_options()
  ("problem", "Initial value that is used in the simulation", cxxopts::value<string>()->default_value("alfven"))
  ("method", "Dynamical low-rank method (either lie or unconventional)", cxxopts::value<string>()->default_value("lie"))
  ("final_time", "Time to which the simulation is run", cxxopts::value<double>()->default_value("1.0"))
  ("deltat", "The time step used in the simulation (usually denoted by \\Delta t or tau)", cxxopts::value<double>()->default_value("1e-5"))
  ("r,rank", "Rank of the simulation", cxxopts::value<int>()->default_value("5"))
  ("n", "Number of grid points (as a whitespace separated list)", cxxopts::value<string>()->default_value("100 100 100 100"))
  ("omp_threads", "Number of OpenMP threads used in CPU parallelization (by default half the number of processes reported by the operating system are used)", cxxopts::value<int>()->default_value("-1"))
  ("kx", "wavenumber in the x-direction", cxxopts::value<double>()->default_value("0.14142135623730950"))
  ("ky", "wavenumber in the y-direction", cxxopts::value<double>()->default_value("0.14142135623730950"))
  ("kz", "wavenumber in the z-direction", cxxopts::value<double>()->default_value("6.2831853071795865"))
  ("Me", "fraction of electron to ion mass", cxxopts::value<double>()->default_value("0.00054644808743169399"))
  ("betaMe", "plasma beta", cxxopts::value<double>()->default_value("4.0"))
  ("rho_i", "ion gyro/Larmor radius", cxxopts::value<double>()->default_value("1.0"))
  ("alpha", "strength of perturbation", cxxopts::value<double>()->default_value("1e-2"))
  ("discretization", "space and time discretization used", cxxopts::value<string>()->default_value("fft"))
  ("h,help", "Help message")
  ;
  auto result = options.parse(argc, argv);

  if(result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  #ifdef __OPENMP__
  int num_threads = result["omp_threads"].as<int>();
  if(num_threads == -1)
    num_threads = omp_get_num_procs()/2;
  omp_set_num_threads(num_threads);

  #pragma omp parallel
  {
    if(omp_get_thread_num()==0){
      cout << "Number of threads: " << omp_get_num_threads() << endl;
    }
  }
  #endif

  double rho_i = result["rho_i"].as<double>();
  double kx = result["kx"].as<double>();
  double ky = result["kx"].as<double>();
  double M_e = result["Me"].as<double>();
  double betaMe = result["betaMe"].as<double>();
  double beta = betaMe*M_e;
  double C_P = 1.0/pow(rho_i,2);
  double C_A = beta/pow(rho_i,2);
  double kpar = result["kz"].as<double>();
  double Vmax = 6.0/sqrt(M_e);
  double alpha = result["alpha"].as<double>();
  //double impl_tol = result["impl_tol"].as<double>();
  discretization discr;
  string str_discr = result["discretization"].as<string>();
  if(str_discr == "fft") {
    discr = discretization::fft;
  } else if(str_discr == "lw") {
    discr = discretization::lw;
  } else {
    cout << "ERROR: Discretization " << str_discr << " is not known." << endl;
    exit(1);
  }


  Index r = result["r"].as<int>();
  mind<4> N = parse<4>(result["n"].as<string>());
  mind<2> N_xx = {N[0], N[1]}; 
  mind<2> N_zv = {N[2], N[3]};
  mfp<4> lim_xx = {0.0,2*M_PI/kx,0.0,2*M_PI/ky};
  //mfp<4> lim_zv = {0.0,2*M_PI,-Vmax,Vmax};
  mfp<4> lim_zv = {0.0,2.0*M_PI/kpar,-Vmax,Vmax};
  grid_info<2> gi(r, N_xx, N_zv, lim_xx, lim_zv, M_e, C_P, C_A,discr); 
//gi.debug_adv_vA = false; // TODO: remove

  double t_final = result["final_time"].as<double>();
  double deltat = result["deltat"].as<double>();
  string method = result["method"].as<string>();

  vec xx1({gi.dxx_mult}), xx2({gi.dxx_mult});
  componentwise_vec_omp(gi.N_xx, [&xx1, &xx2, &gi, kx, ky](Index idx, array<Index,2> i) {
    mfp<2> x  = gi.x(i);
    xx1(idx) = 1.0;
    xx2(idx) = cos(kx*x[0])*cos(ky*x[1]);
  });

  vec vv1({gi.dzv_mult}), vv2({gi.dzv_mult});
  componentwise_vec_omp(gi.N_zv, [&vv1, &vv2, &gi, &alpha, kpar](Index idx, array<Index,2> i) {
    mfp<2> zv  = gi.v(i);
    vv1(idx) = sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
    vv2(idx) = alpha*cos(kpar*zv[0])*sqrt(gi.M_e/M_PI)*exp(-gi.M_e*pow(zv[1],2));
  });
  vector<const double*> X, V;
  X.push_back(xx1.begin());
  X.push_back(xx2.begin());
  V.push_back(vv1.begin());
  V.push_back(vv2.begin());

  integration(method, t_final, deltat, gi, X, V);
}

