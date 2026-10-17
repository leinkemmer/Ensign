# Examples

All examples live in the [`examples/`](https://github.com/leinkemmer/Ensign/tree/master/examples) folder and are built together with the library.
Start with the [minimal example](tutorial.md).

| Program | Description |
|---|---|
| `minimal_dlr_vlasov` | 1+1D Vlasov–Poisson with the projector-splitting integrator; see the [tutorial](tutorial.md) |
| `vlasov_poisson` | 3+3D dynamical low-rank Vlasov–Poisson solver |
| `vlasov_poisson_adapt`, `vlasov_poisson_adapt_reserve` | Rank-adaptive (BUG) 3+3D Vlasov–Poisson solvers, the latter with reduced memory footprint |
| `alfven_waves` | 3+1D solver for Alfvén wave problems |
| `drift_kinetics_interpolatory` | 3+1D drift-kinetic solver using interpolatory low-rank integrators |
| `lhd_instability` | 1+2D lower hybrid drift instability simulation |

The larger examples use [cxxopts](https://github.com/jarro2783/cxxopts); run them with `--help` for the available options.
