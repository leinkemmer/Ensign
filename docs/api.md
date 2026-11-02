# API reference

The API reference is generated from the source code with Doxygen and is available here:

**[Modules](doxygen/modules.html)** · **[Files](doxygen/files.html)** · **[Classes](doxygen/annotated.html)** · **[Functions](doxygen/functions.html)**

## Overview of the code

Ensign's headers provide data structures and operations for dynamical low-rank
(DLR) and interpolatory low-rank methods, along with general-purpose numerical
utilities. Below is an overview of the code structure.

### Fundamental data types

- `generic/storage.hpp` defines `multi_array<T, d>`, a contiguous, column-major
  dense array with number of dimensions determined at compile time. Arrays can reside in
  host (CPU) or device (GPU) memory, and can reserve extra capacity to allow for fast
  rank adaptive schemes.
- `lr/lr.hpp` defines `lr2<T>`, the matrix low-rank representation
  \(f = X S V^T\). The basis matrices `X` and `V` and the small coefficient matrix
  `S` represent a large matrix without storing the full `f`. `lr2` uses the same
  host/device storage choice as `multi_array`.
- `generic/common.hpp` provides shared definitions such as `Index` (`ptrdiff_t`) and `stloc`
  (the host/device location selector).

### Building dynamical low-rank schemes

- `generic/matrix.hpp` provides dense matrix and vector operations,
  matrix-vector products, transposes, SVD, QR, symmetric diagonalization,
  and an LU solver. `Matrix::blas_ops` selects
  CPU or optional GPU execution.
- `lr/coefficients.hpp` defines `coeff` and `integrate` for weighted integrals
  of basis functions, producing the small coefficient matrices needed in the DLR
  K, S, and L steps.
- `lr/lr.hpp` provides higher-level operations on `lr2` representations: initialization,
  weighted orthogonalization, addition, multiplication, truncation, inner
  products and norms.
- `generic/kernels.hpp` declares the CUDA kernels used for GPU-side elementwise
  operations and numerical updates. These are low-level building blocks
  that are available alongside the higher-level matrix and low-rank operations.

### Binary tree tensor networks

- `generic/tree.hpp` defines the binary-tree tensor structure: internal nodes
  hold coefficient tensors that connect their two children, while external
  (leaf) nodes hold basis matrices. Each node also
  stores its incoming rank and coefficient matrix.
- `generic/tensor.hpp` supplies
  the supporting tensor operations, including matricization, tensorization, and
  orthogonalization.

### Interpolatory low-rank schemes

`lr/interpolatory.hpp` provides DEIM-style sampling and reconstruction
operations for interpolatory low-rank representations. `deim_ext` selects
sample points, and `colloquation_to_lr` constructs low-rank factors from
sampled matrix entries.

### Other utilities

| Header | Purpose |
|---|---|
| `generic/utility.hpp` | Various utility functions |
| `generic/fft.hpp` | Batched FFT wrappers; uses FFTW on the CPU and cuFFT when built with CUDA |
| `generic/timer.hpp` | Wall clock timers and named timers for profiling |
| `generic/netcdf.hpp` | NetCDF writer for simulation output |
