#include "drift_kinetics_interpolatory.hpp"
#include <generic/netcdf.hpp>

#include <cxxopts.hpp>


void lr_maxwellian(grid_info& gi, std::function<double(double)> T, mat& U, mat& V, blas_ops& blas) {
  mat f_eq({gi.n_x[0], gi.n_v[0]});  // only as a function of r and v_par only
  for(Index j=0;j<gi.n_v[0];j++) {
    for(Index i=0;i<gi.n_x[0];i++) {
      f_eq(i,j) = exp(-pow(gi.vpar(j),2)/(2.0*gi.Ti(gi.rvar(i))));
    }
  }

  Index m = std::min(gi.n_x[0], gi.n_v[0]);
  mat UU({gi.n_x[0], m}), VV({gi.n_v[0], m});
  vec sigma({m});
  svd(f_eq, UU, VV, sigma, blas);
/*
  cout << "sigma: " << sigma << endl;
  cout << "UU: " << endl;
  for(Index i=0;i<gi.n_x[0];i++)
    cout << UU(i,0) << endl;
  cout << "VV: " << endl;
  for(Index i=0;i<gi.n_v[0];i++)
    cout << VV(i,0) << endl;
*/
  for(Index ir=0;ir<gi.r;ir++) {
    for(Index i=0;i<gi.n_x[0];i++) { 
      U(i, ir) = UU(i, ir)*sigma(ir);
    }
  }
  
  for(Index ir=0;ir<gi.r;ir++) {
    for(Index i=0;i<gi.n_v[0];i++) { 
      V(i, ir) = VV(i, ir);
    }
  }

  double err = 0.0;
  for(Index j=0;j<gi.n_v[0];j++) {
    for(Index i=0;i<gi.n_x[0];i++) {
      double val = 0.0;
      for(Index ir=0;ir<gi.r;ir++) {
        val += U(i, ir)*V(j, ir);
      }
      err = max_err(err, abs(val - f_eq(i,j)));
    }
  }
  cout << "err: " << err << endl;


  cout << "Error of initial value: " << sigma(gi.r) << endl;
}

double electric_energy(const vec& potential, const grid_info& gi) {
  double ee=0.0;
  for(Index k=0;k<gi.n_x[2];k++) {
    for(Index j=0;j<gi.n_x[1];j++) {
      Index idx = gi.lin_idx_x({gi.n_x[0]/2,j,k});
      ee += pow(potential(idx),2)*gi.h_x[1]*gi.h_x[2];
    }
  }

  return sqrt(ee);
}

void save_lr(string fn, const lr2<double>& lr_sol, const vec& potential, const grid_info& gi) {
    nc_writer ncw(fn, {gi.n_x[0], gi.n_x[1], gi.n_x[2], gi.n_v[0], gi.n_v[1], gi.r}, {"rvar", "theta", "phi", "vpar", "mu", "r"});
    ncw.add_var("r", {"r"});
    ncw.add_var("rvar", {"rvar"});
    ncw.add_var("theta", {"theta"});
    ncw.add_var("phi", {"phi"});
    ncw.add_var("vpar", {"vpar"});
    ncw.add_var("mu", {"mu"});
    ncw.add_var("X", {"r", "phi", "theta", "rvar"});
    ncw.add_var("S", {"r", "r"});
    ncw.add_var("V", {"r", "mu", "vpar"});
    ncw.add_var("potential", {"phi", "theta", "rvar"});

    ncw.start_write_mode();

    vector<double> vec_r(gi.r);
    for(Index i=0;i<gi.r;i++)
      vec_r[i] = i;

    vector<double> vec_rvar(gi.n_x[0]);
    for(Index i=0;i<gi.n_x[0];i++)
        vec_rvar[i] = gi.rvar(i);

    vector<double> vec_theta(gi.n_x[1]);
    for(Index i=0;i<gi.n_x[1];i++)
        vec_theta[i] = gi.theta(i);
    
    vector<double> vec_phi(gi.n_x[2]);
    for(Index i=0;i<gi.n_x[2];i++)
        vec_phi[i] = gi.phi(i);

    vector<double> vec_vpar(gi.n_v[0]);
    for(Index i=0;i<gi.n_v[0];i++)
        vec_vpar[i] = gi.vpar(i);

    vector<double> vec_mu(gi.n_v[1]);
    for(Index i=0;i<gi.n_v[1];i++)
        vec_mu[i] = gi.mu(i);

    ncw.write("r", vec_r.data());
    ncw.write("rvar", vec_rvar.data());
    ncw.write("theta", vec_theta.data());
    ncw.write("phi", vec_phi.data());
    ncw.write("vpar", vec_vpar.data());
    ncw.write("mu", vec_mu.data());

    ncw.write("X", lr_sol.X.data());
    ncw.write("S", lr_sol.S.data());
    ncw.write("V", lr_sol.V.data());

    ncw.write("potential", potential.data());
}


int main(int argc, char** argv) {
  cxxopts::Options options("drift_kinetic_interpolatory", "2+2 dimensional dynamical low-rank solver for Alfven wave problems");
  options.add_options()
  ("problem", "Initial value that is used in the simulation", cxxopts::value<string>()->default_value("ITG"))
  ("spaced", "cd2, cd4, or upwind3", cxxopts::value<string>()->default_value("cd2"))
  ("final_time", "Time to which the simulation is run", cxxopts::value<double>()->default_value("6000.0"))
  ("deltat", "The time step used in the simulation (usually denoted by \\Delta t or tau)", cxxopts::value<double>()->default_value("2.0"))
  ("r,rank", "Rank of the simulation", cxxopts::value<int>()->default_value("15"))
  ("rank_oversampling", "The rank for oversampling (must be >= r)", cxxopts::value<int>()->default_value("20"))
  ("n", "Number of grid points (as a whitespace separated list)", cxxopts::value<string>()->default_value("64 31 33 128"))
  ("omp_threads", "Number of OpenMP threads used in CPU parallelization (by default half the number of processes reported by the operating system are used)", cxxopts::value<int>()->default_value("-1"))
  ("snapshots", "Number of files written to disk", cxxopts::value<int>()->default_value("2"))
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
  #endif


  Index r = result["r"].as<int>();
  Index r_over = result["r"].as<int>();
  mind<4> N = parse<4>(result["n"].as<string>());
  mind<3> n_x = {N[0], N[1], N[2]};
  mind<2> n_v = {N[3], 1};
  double final_time = result["final_time"].as<double>();
  double deltat = result["deltat"].as<double>();
  Index snapshots = result["snapshots"].as<int>();
  string spaced = result["spaced"].as<string>();

  // set the domain (TODO: Is R0 consistent with that?)
  double r_min=0.1, r_max = 14.5, r_p = 0.5*(r_min+r_max);
  mfp<5> lim_a = {r_min, 0.0, 0.0, -9.0, 0.0};
  mfp<5> lim_b = {r_max, 2.0*M_PI, 2.0*M_PI, 9.0, 1.0};
  grid_info gi(r, r_over, n_x, n_v, lim_a, lim_b, 239.808, 1.0, 1.0, 1.0, 1.0);

  // compute the normalization factor for n0
  double kappa_n0 = 0.055;
  double delta_n0 = 2.0*1.45;
  gi.n0 = [kappa_n0, delta_n0, r_p](double r) { return exp(-kappa_n0*delta_n0*tanh((r-r_p)/delta_n0)); };
  double norm_n0 = 0.0;
  for(Index i=0;i<gi.n_x[0];i++) {
      double rvar = gi.rvar(i);
      norm_n0 += gi.n0(rvar)*gi.h_x[0];
  }
  norm_n0 = (r_max - r_min)/norm_n0;
  gi.n0 = [norm_n0, kappa_n0, delta_n0, r_p](double r) { return norm_n0*exp(-kappa_n0*delta_n0*tanh((r-r_p)/delta_n0)); };

  double kappa_Te = 0.27586, kappa_Ti = 0.27586;
  double delta_Te = 1.45, delta_Ti = 1.45;
  gi.Te = [kappa_Te, delta_Te, r_p](double r) { return exp(-kappa_Te*delta_Te*tanh((r-r_p)/delta_Te)); };
  gi.Ti = [kappa_Ti, delta_Ti, r_p](double r) { return exp(-kappa_Ti*delta_Ti*tanh((r-r_p)/delta_Ti)); }; // TODO: does that need to be part of gi?


  blas_ops blas;

  // set the initial condition
  double eps = 1e-6;
  double mode_n=1, mode_m=5;
  double delta_r = 8.0;


  mat UU({gi.n_x[0], gi.r}), VV({gi.n_v[0], gi.r});
  lr_maxwellian(gi, gi.Ti, UU, VV, blas);

  vector<const double*> X;
  mat x1({gi.N_x, gi.r});
  for(Index ir=0;ir<gi.r;ir++) {
    for(Index k=0;k<gi.n_x[2];k++) {
      for(Index j=0;j<gi.n_x[1];j++) {
        for(Index i=0;i<gi.n_x[0];i++) {
          Index idx = gi.lin_idx_x({i,j,k});
          double rvar = gi.rvar(i);
          double theta = gi.theta(j);
          double phi = gi.phi(k);

          x1(idx, ir) = UU(i, ir)*gi.n0(rvar)/sqrt(2.0*M_PI*gi.Ti(rvar))*(1.0 + eps*exp(-pow(rvar-r_p,2)/delta_r)*cos(mode_n*phi + mode_m*theta));
        }
      }
    }

    X.push_back(x1.begin()+gi.N_x*ir);
  }
 
  vector<const double*> V;
  mat v1({gi.N_v, gi.r});
  for(Index ir=0;ir<gi.r;ir++) {
    for(Index i=0;i<gi.n_v[0];i++) {
      v1(i, ir) = VV(i, ir); 
    } 

    V.push_back(v1.begin()+gi.N_v*ir);
  }


  lr2<double> f(gi.r, {gi.N_x, gi.N_v});
  initialize(f, X, V, 1.0, 1.0, blas);

  vec potential({gi.N_x});
  mat L({gi.N_v, gi.r});
  quasi_neutrality_solver qns(gi);

  double t = 0.0;
  bool final_step = false;
  Index n = 0;
  Index num_steps = int(ceil(final_time/deltat));
  ofstream fs("evolution.data");
  while(t<final_time && !final_step) {
      gt::start("timestep");

      if(final_time - t < deltat) {
        deltat = final_time - t;
        final_step = true;
      }
  
      gt::start("qn_rhs");
      blas.matmul_transb(f.V, f.S, L);
      qns.compute_rhs(f.X, L);
      gt::stop("qn_rhs");

      //cout << qns.rhs << endl;
      //exit(1);
      gt::start("qn_solve");
      qns.solve(potential);
      gt::stop("qn_solve");

      if(snapshots>=2 && (n==0 || (n % int(ceil(num_steps/double(snapshots-1))) == 0))) {
        std::stringstream ss_fn;
        ss_fn << "out-" << t << ".nc";
        save_lr(ss_fn.str(), f, potential, gi);
      }

      double ee = electric_energy(potential, gi);

      cout << "\r" << std::setw(30) << "";
      cout << "\rt=" << t << endl;
      
      fs << t << "\t" << ee << endl;

      gt::start("rk4");
      if(spaced == "cd2") {
        rk4(deltat, gi, f, rhs_driftkinetic_cd2, potential, blas);
      } else if(spaced == "cd4") {
        rk4(deltat, gi, f, rhs_driftkinetic_cd4, potential, blas);
      } else if(spaced == "upwind3") {
        rk4(deltat, gi, f, rhs_driftkinetic_upwind3, potential, blas);
      } else {
        cout << "ERROR: space discretization " << spaced << " not found." << endl;
        exit(1);
      }
      gt::stop("rk4");

      t += deltat;
      n++;

      gt::stop("timestep");
    }
        
    std::stringstream ss_fn;
    ss_fn << "out-" << t << ".nc";
    save_lr(ss_fn.str(), f, potential, gi);

  cout << gt::sorted_output() << endl;
}