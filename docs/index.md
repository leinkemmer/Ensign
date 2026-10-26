# Ensign

**Ensign** is a C++ framework for the efficient implementation of
dynamical low-rank and interpolatory low-rank algorithms on multi-core CPUs and GPUs (CUDA).

High-dimensional problems, such as the six-dimensional Vlasov equation of plasma physics, are prohibitively expensive to solve on a full grid. The dynamical low-rank method approximates the solution by a low-rank factorization and evolves the factors in time, which reduces both memory and run time by orders of magnitude (see the review [[3]](references.md#ref3)). Ensign provides the building blocks that make implementing such schemes easy.

Ensign has also been used to solve very high-dimensional chemical reaction networks in the [atropy software](https://atropy.gitlab.io/) using binary tree tensor networks.

Ensign is developed by [Lukas Einkemmer](https://www.einkemmer.net/) [Fabio Cassini](https://sites.google.com/view/fabiocassini/), [Julian Mangott](https://scholar.google.com/citations?user=UvfaGaIAAAAJ), and Stefan Brunner.


## Features

- Low-rank data structures (`lr2`) for matrix factorization and binary tree-tensor networks (`node`)
- Support for interpolatory low-rank integrators based on DEIM sampling.
- Building blocks: orthogonalization (Gram–Schmidt / Householder with weighted inner products),
  computation of low-rank coefficients (integrals of basis functions), FFT wrappers, matrix operations.
- Identical code paths on CPU (OpenBLAS or Intel MKL, OpenMP) and GPU (CUDA).
- Examples ranging from a minimal 1+1D Vlasov–Poisson solver to full 3+3D simulations.


## Where to go next

- [Installation](installation.md)
- [Tutorial: a minimal low-rank Vlasov–Poisson solver](tutorial.md)
- [API reference](api.md)

