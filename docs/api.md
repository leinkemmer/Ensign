# API reference

The API reference is generated from the source code with Doxygen and is available here:

**[Classes and structs](doxygen/annotated.html)** · **[Functions and members](doxygen/functions.html)** · **[Files](doxygen/files.html)**

## Overview of the headers

| Header | Contents |
|---|---|
| `generic/storage.hpp` | `multi_array<T,d>`: dense multi-dimensional array on host or device |
| `generic/matrix.hpp` | `blas_ops` (matrix products), point-wise operations, SVD, diagonalization, LU solver |
| `generic/fft.hpp` | FFT wrappers (FFTW on CPU, cuFFT on GPU) |
| `generic/tensor.hpp`, `generic/tree.hpp` | Tensor and tree-tensor-network data structures |
| `lr/lr.hpp` | `lr2`: low-rank factorization \(X S V^T\), `initialize`, `orthogonalize`, addition, truncation, inner products |
| `lr/coefficients.hpp` | `coeff`, `integrate`: computation of the low-rank coefficients (integrals of products of basis functions) |
| `lr/interpolatory.hpp` | Interpolatory (adaptive cross approximation type) low-rank integrators |

