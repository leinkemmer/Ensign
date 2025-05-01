
#include "lhd_instability.hpp"

#include <cxxopts.hpp>
#include <iostream>


int main(int argc, char** argv) {

  cxxopts::Options options("lhd-instability", "1+2 dimensional dynamical low-rank solver for acceleration-driven lower hybrid drift instability");
  options.add_options()
  ("problem", "Initial value that is used in the simulation", cxxopts::value<string>()->default_value("default"))
  ("final_time", "Time to which the simulation is run", cxxopts::value<double>()->default_value("400.0"))
  ("deltat", "The time step used in the simulation (usually denoted by \\Delta t or tau)", cxxopts::value<double>()->default_value("8e-4"))
  ("r_i,rank_ions", "Rank of the ions", cxxopts::value<int>()->default_value("10"))
  ("r_e,rank_electrons", "Rank of the electrons", cxxopts::value<int>()->default_value("10"))
  ("n", "Number of grid points (as a whitespace separated list)", cxxopts::value<string>()->default_value("1152 512 512"))
  ("snapshots", "Number of files written to disk", cxxopts::value<int>()->default_value("0"))
  ("h,help", "Help message")
  ;
  auto result = options.parse(argc, argv);

  if(result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  // parse the command line parameters
  Index r_e = result["r_e"].as<int>();
  Index r_i = result["r_i"].as<int>();
  mind<3> N = parse<3>(result["n"].as<string>());
  Index snapshots = result["snapshots"].as<int>();

  // parameters for the simulation
  Index n_x = N[0]; 
  mind<2> n_v = {N[1], N[2]};
  //mfp<3> a_e = {0.0, -1.125, -1.128};
  //mfp<3> b_e = {1.414, 1.125, 1.122};
  //mfp<3> a_i = {0.0, -0.225, -0.15};
  //mfp<3> b_i = {1.414, 0.225, 0.30};
  // smaller domain
  mfp<3> a_e = {0.0, -0.5*1.125, -0.5*1.128};
  mfp<3> b_e = {1.414, 0.5*1.125, 0.5*1.122};
  mfp<3> a_i = {0.0, -0.5*0.225, -0.0375};
  mfp<3> b_i = {1.414, 0.5*0.225, 0.1875};
  double g = -7.5e-3;
  double B = 1.0;
  double m_i = 1.0;
  double m_e = 1.0/25.0;
  double Omega_e = 2.5;
  double Omega_i = 0.1;
  double alpha = 2e-4;
  double T_e = 6.25e-4;
  double T_i = T_e;

  double t_final = result["final_time"].as<double>();
  double deltat = result["deltat"].as<double>();

  grid_info gi_e(r_e, n_x, n_v, a_e, b_e, g, B, -1.0, m_e, Omega_i);
  grid_info gi_i(r_i, n_x, n_v, a_i, b_i, g, B, 1.0, m_i, Omega_i);

  // setup initial condition
  vec xx1_e({n_x}), xx1_i({n_x});
  for(Index i=0;i<n_x;i++) {
    double y  = gi_e.x(i);
    double p = abs(y-0.5*b_e[0]) <= 1e-14 ? 1.0 : sin(86.6262*(y-0.5*b_e[0]))/(86.6262*(y-0.5*b_e[0]));
    xx1_e(i) = 1.0+alpha*p;
    xx1_i(i) = 1.0+alpha*p;
  }

  vec vv1_e({gi_e.N_v}), vv1_i({gi_i.N_v});
  componentwise_vec_omp(gi_e.n_v, [&vv1_e, &vv1_i, &gi_e, &gi_i, &T_e, &T_i, &Omega_e, &Omega_i](Index idx, array<Index,2> i) {
    mfp<2> v_e  = gi_e.v(i);
    vv1_e(idx) = gi_e.m/(2.0*M_PI*T_e)*exp(-gi_e.m/(2.0*T_e)*(pow(v_e[1]-gi_e.g/Omega_e,2)+pow(v_e[0],2)));
    mfp<2> v_i  = gi_i.v(i);
    vv1_i(idx) = gi_i.m/(2.0*M_PI*T_i)*exp(-gi_i.m/(2.0*T_i)*(pow(v_i[1]+gi_i.g/Omega_i,2)+pow(v_i[0],2)));
  });

  vector<const double*> X_e, V_e;
  X_e.push_back(xx1_e.begin());
  V_e.push_back(vv1_e.begin());
  
  vector<const double*> X_i, V_i;
  X_i.push_back(xx1_i.begin());
  V_i.push_back(vv1_i.begin());


  // runt the simulation
  vlasov vlasov_e(gi_e, X_e, V_e);
  vlasov vlasov_i(gi_i, X_i, V_i);
  poisson poi(gi_e, gi_i);
  vec n_e({gi_e.n_x}), hx_e({gi_e.n_x}), hy_e({gi_e.n_x});
  vec n_i({gi_i.n_x}), hx_i({gi_i.n_x}), hy_i({gi_i.n_x});

  if(snapshots >= 2) {
    save_lr("f_e_0.nc", vlasov_e.f, gi_e);
    save_lr("f_i_0.nc", vlasov_i.f, gi_i);
  }

  std::ofstream fs("evolution.data");
  fs << "# electric_energy nu mass_err_electrons mass_err_ions mom_err_electrons mom_err_ions energy_err kinetic_energy_electrons kinetic_energy_ions sum_g_energy_electrons sum_g_energy_ions" << endl;
  double ken_e_0 = vlasov_e.compute_nh(n_e, hx_e, hy_e);
  double ken_i_0 = vlasov_i.compute_nh(n_i, hx_i, hy_i);
  double mass_e_0 = integrate_x(n_e, gi_e);
  double mass_i_0 = integrate_x(n_i, gi_i);
  double mom_e_0 = integrate_x(hy_e, gi_e);
  double mom_i_0 = integrate_x(hy_i, gi_i);
  poi.compute(n_e, n_i);
  double energy_0 = ken_e_0 + ken_i_0 + poi.ee;
  double sum_g_energy_e = 0.0, sum_g_energy_i = 0.0;

  double t = 0.0;
  Index num_steps = ceil(t_final/deltat);
  gt::start("simulation");
  for(Index n=0;n<num_steps;n++) {
    if(t_final -t < deltat)
      deltat = t_final - t;

    gt::start("diagnostics");
    double ken_e = vlasov_e.compute_nh(n_e, hx_e, hy_e);
    double ken_i = vlasov_i.compute_nh(n_i, hx_i, hy_i);
    poi.compute(n_e, n_i);
    double nu = poi.compute_anomcoll(n_e, n_i, hy_e, hy_i);
    double mass_e = integrate_x(n_e, gi_e);
    double mass_i = integrate_x(n_i, gi_i);
    double mom_e = integrate_x(hy_e, gi_e);
    double mom_i = integrate_x(hy_i, gi_i);
    sum_g_energy_e -= deltat*gi_e.g*gi_e.m*integrate_x(hx_e, gi_e);
    sum_g_energy_i -= deltat*gi_i.g*gi_i.m*integrate_x(hx_i, gi_i);
    double energy = ken_e + ken_i + poi.ee + sum_g_energy_e + sum_g_energy_i;
    gt::stop("diagnostics");
    fs << t << "\t\t" << poi.ee << "\t\t" << nu << "\t\t" << abs(mass_e-mass_e_0)/mass_e_0 << "\t\t" << abs(mass_i-mass_i_0)/mass_i_0 << "\t\t" << abs(mom_e-mom_e_0) << "\t\t" << abs(mom_i-mom_i_0) << "\t\t" << abs(energy-energy_0)/abs(energy_0) << " " << ken_e << " " << ken_i << " " << sum_g_energy_e << " " << sum_g_energy_i << endl;

    gt::start("step");
    vlasov_e.step(deltat, poi.E);
    vlasov_i.step(deltat, poi.E);
    gt::stop("step");

    t += deltat;
  
    if(snapshots>2 && n != 0 && ((n+1) % int(ceil(num_steps/double(snapshots-1))) == 0)) {
      std::stringstream ss_e, ss_i;
      ss_e << "f_e_" << t << ".nc";
      ss_i << "f_i_" << t << ".nc";
      save_lr(ss_e.str(), vlasov_e.f, gi_e);
      save_lr(ss_i.str(), vlasov_i.f, gi_i);
    }
  }
  gt::stop("simulation");

  if(snapshots >= 2) {
    std::stringstream ss_e, ss_i;
    ss_e << "f_e_" << t_final << ".nc";
    ss_i << "f_i_" << t_final << ".nc";
    save_lr(ss_e.str(), vlasov_e.f, gi_e);
    save_lr(ss_i.str(), vlasov_i.f, gi_i);
  }

  cout << gt::sorted_output() << endl;
}

