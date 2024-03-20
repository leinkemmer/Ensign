# Readme #

The **Ensign** software framework facilitates the efficient implementation of dynamical low-rank algorithms both on multi-core CPU and GPUs. It provides many primitives to make the implementation of such schemes easier and can work easily with both methods based on the projector splitting and the unconventional integrator. For more details on the dynamical low-rank method and in particular the problems implemented in the examples folder see

[F. Cassini, L. Einkemmer. Efficient 6D Vlasov simulation using the dynamical low-rank framework Ensign.](https://arxiv.org/abs/2110.13481)

First clone the repository

    git clone https://github.com/leinkemmer/Ensign.git

To build the software the following are needed
- CMake
- C++11 compatible C++ compiler
- Fortran compiler (if the included OpenBLAS is used)
- CUDA (if GPU support is desired)
- NetCDF (optinal, required for writing snapshots to disk)

To build the example programs and tests execute

    mkdir build
    cd build
    cmake -DCUDA_ENABLED=ON ..
    make

If you prefer to use Intel MKL as the BLAS and LAPACK backend set

    mkdir build
    cd build
    export MKLROOT=/opt/intel/mkl
    cmake -DMKL_ENABLED=ON -DCUDA_ENABLED=ON ..
    make

## Build instructions on MacOS

If you are on a Mac, you may have to use a different compiler than the system C++ compiler,
since the `clang` that comes installed on OS X does not support OpenMP.
To resolve this, you can install `gcc` via brew:

    brew install gcc

Then, when building Ensign, use the Homebrew `gcc`:

    export CC=/opt/homebrew/opt/gcc/bin/gcc-13

You may have to change the version from 13 to whichever version of gcc Homebrew installed.
