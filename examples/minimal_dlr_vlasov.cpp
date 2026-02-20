#include <cmath>
#include <generic/storage.hpp>
#include <generic/matrix.hpp>
#include <generic/fft.hpp>
#include <lr/lr.hpp>
#include <lr/coefficients.hpp>

using vec  = Ensign::multi_array<double,1>;
using cvec = Ensign::multi_array<complex<double>,1>;
using mat  = Ensign::multi_array<double,2>;
using lr  = Ensign::lr2<double>;

Index nx=128;
Index nv=129;
Index r=5;

double L = 4*M_PI;
double V = 6.0;
double t_final = 40.0;
double deltat = 1e-2;

double hx = L/double(nx);
double hv = 2.0*V/double(nv);

vec vs({nv});
Ensign::Matrix::blas_ops blas;
Ensign::orthogonalize ortho(&blas);


double x(Index i) {
  return i*hx;
}

double v(Index j) {
  return -V + j*hv;
}


vec compute_rho(const lr& f) {
    mat K({nx, r});
    vec int_V({r}), rho({nx});
    
    Ensign::integrate(f.V,-hv,int_V,blas);
    blas.matmul(f.X,f.S,K);
    blas.matvec(K,int_V,rho);
    return rho;
}

vec compute_E(const lr& f) {
    vec rho = compute_rho(f);
    rho += 1.0;

    cvec Ehat({nx/2+1});
    Ensign::fft1d<1> fft(array<Index,1>({nx}), rho, Ehat);
    fft.forward(rho, Ehat);
    for(Index k=1;k<nx/2+1;k++) {
      complex<double> lambda = complex<double>(0.0,2.0*M_PI/L*k);
      Ehat(k) /= lambda*double(nx);
    }
    Ehat(0) = 0.0;
    vec E({nx});
    fft.backward(Ehat, E);
    return E;
}

double compute_ee(const vec& E) {
    double ee = 0.0;
    for(Index i=0;i<nx;i++) {
        ee += 0.5*hx*pow(E(i),2);
    }
    return ee;
}


void rk4(double tau, mat& U, std::function<void(const mat&, mat&)> rhs) {
  mat k1(U), k2(U), k3(U), k4(U), tmp(U), in(U);

  // k1
  rhs(in, k1);

  // k2
  tmp = k1;
  tmp *= 0.5*tau;
  tmp += in;
  rhs(tmp, k2);

  // k3
  tmp = k2;
  tmp *= 0.5*tau;
  tmp += in;
  rhs(tmp, k3);

  // k4
  tmp = k3;
  tmp *= tau;
  tmp += in;
  rhs(tmp, k4);
  
  k1 *= 1.0/6.0*tau;
  U += k1;
  k2 *= 1.0/6.0*tau*2.0; 
  U += k2;
  k3 *= 1.0/6.0*tau*2.0;
  U += k3;
  k4 *= 1.0/6.0*tau;
  U += k4;
}

mat derivative(const mat& M, Index n, double h) {
  mat Mout({n,r});
  for(Index k=0;k<r;k++) {
    for(Index i=0;i<n;i++) {
      Mout(i,k) = (M((i+1)%n, k) - M((i-1+n)%n, k))/(2.0*h);
    }
  }
  return Mout;
}



mat compute_c1(const mat& V) {
  mat Vtmp({nv,r}), c1({r,r});
  Ensign::Matrix::ptw_mult_row(V,vs,Vtmp); // multiply by v
  Ensign::coeff(V, Vtmp, hv, c1, blas);
  return c1;
}

mat compute_c2(const mat& V) {
  mat c2({r,r});
  mat Vv = derivative(V, nv, hv);
  Ensign::coeff(V, Vv, hv, c2, blas);
  return c2;
}

mat compute_d1(const mat& X, const vec& E) {
  mat d1({r,r}), Xtmp({nx,r});
  Ensign::Matrix::ptw_mult_row(X, E, Xtmp);
  Ensign::coeff(X, Xtmp, hx, d1, blas);
  return d1;
}

mat compute_d2(const mat& X) {
  mat d2({r,r});
  mat X_x = derivative(X, nx, hx);
  Ensign::coeff(X, X_x, hx, d2, blas);
  return d2;
}


mat rhs_K(const mat& K, const mat& c1, const mat& c2, const vec& E) {
  mat Kout({nx,r});
  mat Kx = derivative(K, nx, hx);
  blas.matmul_transb(Kx, c1, Kout);
  Kout *= -1.0;

  mat Ktmp({nx,r});
  Ensign::Matrix::ptw_mult_row(K, E, Kx); // use Kx as tmp here
  blas.matmul_transb(Kx, c2, Ktmp);
  Kout += Ktmp;

  return Kout;
}

mat rhs_S(const mat& S, const mat& c1, const mat& c2, const mat& d1, const mat& d2) {
  mat Sout({r,r}), Stmp({r,r}), Stmp2({r,r});
  blas.matmul(d2, S, Stmp);
  blas.matmul_transb(Stmp, c1, Sout);

  blas.matmul(d1, S, Stmp);
  blas.matmul_transb(Stmp, c2, Stmp2);
  Sout -= Stmp2;

  return Sout;
}

mat rhs_L(const mat& L, const mat& d1, const mat& d2) {
  mat Lout({nv,r}), Lptv({nv,r});
  Ensign::Matrix::ptw_mult_row(L, vs, Lptv);
  blas.matmul_transb(Lptv, d2, Lout);
  Lout *= -1.0;

  mat Lv = derivative(L, nv, hv);
  blas.matmul_transb(Lv, d1, Lptv); // use Lptv as a tmp here
  Lout += Lptv;

  return Lout;
}

void time_step(double deltat, lr& f, const vec& E) {
  // K step
  mat K({nx,r});
  mat c1 = compute_c1(f.V);
  mat c2 = compute_c2(f.V);
  blas.matmul(f.X, f.S, K);
  rk4(deltat, K, [&c1, &c2, &E](const mat& K, mat& Kout) {
    Kout = rhs_K(K, c1, c2, E);
  });

  // orthogonalize K to obtain X^{n+1} and S for the next step
  f.X = K;
  ortho(f.X, f.S, hx);

  // S step
  mat d1 = compute_d1(f.X, E);
  mat d2 = compute_d2(f.X);
  rk4(deltat, f.S, [&c1,&c2,&d1,&d2](const mat& S, mat& Sout) {
    Sout = rhs_S(S, c1, c2, d1, d2);
  });

  // L step
  mat L({nv,r});
  blas.matmul_transb(f.V,f.S,L);
  rk4(deltat, L, [&d1, &d2](const mat& L, mat& Lout) {
    Lout = rhs_L(L, d1, d2);
  });

  // orthogonalize L to obtain V^{n+1} and S^{n+1}
  f.V = L;
  ortho(f.V, f.S, hv);
}


int main() {
  #ifdef __OPENMP__
  // To avoid oversubscription (which is the default on many systems)
  omp_set_num_threads(1);
  #endif

  // Setup initial value
  vec init_x({nx});
  for(Index i=0;i<nx;i++) {
      init_x(i) = 1.0 + 1e-2*cos(0.5*x(i));
  }

  vec init_v({nv});
  for(Index j=0;j<nv;j++) {
    init_v(j) = exp(-0.5*pow(v(j),2))/sqrt(2.0*M_PI);
  }

  // Initializes a rank 1 initial value given by init_x(x)*init_v(v)
  // with the standard L^2 inner product
  lr f(r, {nx, nv});
  vector<const double*> in_x, in_v;
  in_x.push_back(init_x.begin());
  in_v.push_back(init_v.begin());
  Ensign::initialize(f, in_x, in_v, hx, hv, blas);

  for(Index j=0;j<nv;j++) {
    vs(j) = v(j);
  }

  std::ofstream fs("evolution.data");
  fs << "# t electric_energy" << std::endl;;

  double t = 0.0;
  Index num_steps = Index(ceil(t_final/deltat));
  for(Index i=0;i<num_steps;i++) {
    if(t_final - t < deltat) {
      deltat = t_final - t;
    }

    vec E = compute_E(f);
    double ee = compute_ee(E);
    fs << t << "\t\t" << ee << endl;

    time_step(deltat, f, E);

    t += deltat;
    std::cout << '\r';
    std::cout << "t=" << t;
  }
}
