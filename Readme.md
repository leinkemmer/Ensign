# Readme

The **Ensign** software framework facilitates the efficient implementation of dynamical low-rank algorithms both on multi-core CPU and GPUs. It provides many primitives to make the implementation of such schemes easier and can work easily with both methods based on the projector splitting and the unconventional integrator. For more details on the dynamical low-rank method and in particular the problems implemented in the examples folder see

[F. Cassini, L. Einkemmer. Efficient 6D Vlasov simulation using the dynamical low-rank framework Ensign.](https://arxiv.org/abs/2110.13481)

First clone the repository

    git clone https://github.com/leinkemmer/Ensign.git

To build the software the following are needed
- CMake
- C++11 compatible C++ compiler
- Fortran compiler (if the included OpenBLAS is used)
- CUDA (if GPU support is desired)

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

## MacOS

### OpenBLAS
If OpenBLAS is used as a BLAS backend, a Fortran compiler has to be installed. Since Apple Clang, the native compiler collection of MacOS, does not ship a Fortran compiler, one has to install a Fortran compiler manually. To obtain `gfortran-14`, the Fortran compiler of GCC, install `gcc-14` via brew (see also next section).

If CMake does not find the Fortran compiler automatically, you have to set the `FC` environment variable accordingly:

    export FC=/path/to/fortran/compiler

Additionally, you might also have to set the CMake cache entry `CMAKE_Fortran_COMPILER` to the full path of the Fortran compiler.

#### OpenMP
Moreover, Apple Clang does not officially support OpenMP. Therefore, you have to use instead a different compiler collection, for example GCC. Install `gcc-14` again via brew:

    brew install gcc@14

Invoke the brew command

    brew info gcc@14

to find the installation path of the C, C++ and Fortran compilers of `gcc-14`. With this installation path, set the `CC`, `CXX` and `FC` (use `gfortran-14`, when OpenBLAS is used) environment variables to use the Homebrew `gcc-14` when building Ensign:

    export CC=/path/to/gcc-14
    export CXX=/path/to/g++-14
    export FC=/path/to/gfortran-14
