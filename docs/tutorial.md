# Tutorial: a minimal low-rank Vlasov–Poisson solver

This tutorial walks through [`examples/minimal_dlr_vlasov.cpp`](https://github.com/leinkemmer/Ensign/blob/master/examples/minimal_dlr_vlasov.cpp),
a complete 1+1D dynamical low-rank solver in about 200 lines.
After building (see [Installation](installation.md)) run

    ./build/examples/minimal_dlr_vlasov

It writes the electric energy as a function of time to `evolution.data`.

## The problem

We solve the Vlasov–Poisson equation for the electron distribution function \(f(t,x,v)\),

\[
\partial_t f + v\,\partial_x f - E(t,x)\,\partial_v f = 0, \qquad
\partial_x E = 1 - \int f \,dv ,
\]

on a periodic domain \(x\in[0,L)\), \(L=4\pi\), and \(v\in[-V,V]\). The initial value
\(f_0 = (1+10^{-2}\cos(x/2))\,\mathrm{e}^{-v^2/2}/\sqrt{2\pi}\) leads to Landau damping of the electric energy.

## The dynamical low-rank approximation

Instead of storing \(f\) on the full \(n_x\times n_v\) grid, we approximate it by a rank-\(r\) factorization

\[
f(t,x,v) \approx \sum_{i,j=1}^r X_i(x)\, S_{ij}(t)\, V_j(v),
\]

where the \(X_i\) and \(V_j\) are orthonormal bases and \(S\) is an \(r\times r\) coefficient matrix.
The evolution equations for \(X\), \(S\), \(V\) are derived by projecting the right-hand side onto the tangent space
of the low-rank manifold (see the review [[3]](references.md#ref3)). Ensign uses the **projector-splitting integrator**, which
solves three sub-problems in sequence (see [[2]](references.md#ref2) for more details):

1. **K step:** update \(K = XS\) with \(V\) fixed, then orthogonalize \(K = X^{n+1} S\) (QR).
2. **S step:** update \(S\) with \(X^{n+1}\) and \(V\) fixed (integrated *backward* in time).
3. **L step:** update \(L = VS^T\) with \(X^{n+1}\) fixed, then orthogonalize \(L = V^{n+1}S^{n+1\,T}\).

Each sub-problem only involves the one-dimensional functions and a few small \(r\times r\) coefficient matrices:

\[
c^1_{ij}=\int V_i\, v\, V_j\,dv,\quad c^2_{ij}=\int V_i\,\partial_v V_j\,dv,\quad
d^1_{ij}=\int X_i\, E\, X_j\,dx,\quad d^2_{ij}=\int X_i\,\partial_x X_j\,dx .
\]

## Walkthrough

### Setup and data types

```cpp
using vec  = Ensign::multi_array<double,1>;
using mat  = Ensign::multi_array<double,2>;
using lr   = Ensign::lr2<double>;
```

[`multi_array`](api.md) is Ensign's dense array type (it can live on CPU or GPU, here on the CPU).
`lr2<double>` stores the three factors `X` (\(n_x\times r\)), `S` (\(r\times r\)) and `V` (\(n_v\times r\)).
The global objects

```cpp
Ensign::Matrix::blas_ops blas;
Ensign::orthogonalize ortho(&blas);
```

provide the BLAS/LAPACK-based matrix operations (`matmul`, `matmul_transb`, ...) and the QR
decomposition used in the K and L steps.

### Initial value

A rank-1 initial value \(f_0 = X_1(x)V_1(v)\) is given as plain arrays and handed to `Ensign::initialize`:

```cpp
lr f(r, {nx, nv});
vector<const double*> in_x, in_v;
in_x.push_back(init_x.begin());
in_v.push_back(init_v.begin());
Ensign::initialize(f, in_x, in_v, hx, hv, blas);
```

`initialize` orthonormalizes the provided functions with respect to the \(L^2\) inner product with grid weights `hx`, `hv`
and fills the remaining \(r-1\) basis functions with random orthogonal functions.

### Density and electric field

The density \(\rho=\int f\,dv\) is computed directly from the low-rank factors,
\(\rho = X S \,(\int V\,dv)\), at a cost of \(O(r(n_x+n_v))\) instead of \(O(n_xn_v)\):

```cpp
Ensign::integrate(f.V, -hv, int_V, blas);   // int_V_j = -∫ V_j dv
blas.matmul(f.X, f.S, K);
blas.matvec(K, int_V, rho);
```

(the negative weight accounts for the sign in the Poisson equation, so `rho` holds \(-\int f\,dv\); adding the background `rho += 1.0` gives the right-hand side \(1-\int f\,dv\)).
`compute_E` then solves \(\partial_xE=1-\int f\,dv\) with the FFT wrapper `Ensign::fft1d` by dividing by \(ik\) in Fourier space.

### Coefficients

The coefficient matrices are computed with `Ensign::coeff`, which approximates \(\int a_i b_j\) by a quadrature rule with weight `h`:

```cpp
mat compute_c1(const mat& V) {
  mat Vtmp({nv,r}), c1({r,r});
  Ensign::Matrix::ptw_mult_row(V, vs, Vtmp); // multiply by v
  Ensign::coeff(V, Vtmp, hv, c1, blas);
  return c1;
}
```

`ptw_mult_row` multiplies each column by a grid function (here \(v\)). Derivatives are approximated with
second-order central finite differences (`derivative`). The other coefficients `c2`, `d1`, `d2` are handled analogously.

### Right-hand sides and time stepping

Each substep is a matrix ODE that is integrated with a classical 4th-order Runge–Kutta method (`rk4`). The right-hand sides are expressed through matrix products:

```cpp
// K' = -K_x c1^T + (E K) c2^T
mat rhs_K(const mat& K, const mat& c1, const mat& c2, const vec& E);
// S' = d2 S c1^T - d1 S c2^T   (backward in time)
mat rhs_S(const mat& S, const mat& c1, const mat& c2, const mat& d1, const mat& d2);
// L' = -v L d2^T + L_v d1^T
mat rhs_L(const mat& L, const mat& d1, const mat& d2);
```

### One time step

`time_step` ties everything together:

```cpp
// K step
blas.matmul(f.X, f.S, K);
rk4(deltat, K, ...);
f.X = K;
ortho(f.X, f.S, hx);      // K = X^{n+1} S

// S step
d1 = compute_d1(f.X, E);  d2 = compute_d2(f.X);
rk4(deltat, f.S, ...);

// L step
blas.matmul_transb(f.V, f.S, L);
rk4(deltat, L, ...);
f.V = L;
ortho(f.V, f.S, hv);      // L = V^{n+1} S^{n+1 T}
```

`ortho(Q, R, w)` orthonormalizes the columns of `Q` with respect to the weighted inner product (weight `w`),
overwrites `Q` with the orthonormal basis and stores the triangular factor in `R`.
Note that the electric field `E` is computed once per time step from the state at the beginning of the step and is kept fixed during the substeps.

### Main loop

The main loop computes the electric field and energy \(\tfrac12\int E^2dx\), writes it to `evolution.data` and advances
the solution until `t_final`:

```cpp
for(Index i=0;i<num_steps;i++) {
  vec E = compute_E(f);
  fs << t << "\t\t" << compute_ee(E) << endl;
  time_step(deltat, f, E);
  t += deltat;
}
```

## Results

Plot the second column of `evolution.data` on a logarithmic scale; the electric energy decays
with the Landau damping rate (oscillating). For example with gnuplot:

    gnuplot -persist -e "set logscale y; plot 'evolution.data' w l"

Try different ranks `r`, grid sizes and time step sizes at the top of the file to see how the low-rank approximation works.

## Going further

- The [other examples](examples.md) extend this approach to 3+3 dimensions, GPUs, and rank adaptivity, etc.
- The full list of building blocks is in the [API reference](api.md).
- The design and performance of Ensign are discussed in [[1]](references.md#ref1).
