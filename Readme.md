# Readme #

The **Ensign** software framework facilitates the efficient implementation of dynamical low-rank algorithms both on multi-core CPU and GPUs. It provides many primitives to make the implementation of such schemes easier and can work easily with both methods based on the projector splitting and the unconventional integrator. For more details on the dynamical low-rank method and in particular the problems implemented in the examples folder see

[F. Cassini, L. Einkemmer. Efficient 6D Vlasov simulation using the dynamical low-rank framework Ensign.](https://arxiv.org/abs/2110.13481)

The repository makes use of git submodules. The easiest way to obtain all the relevant sources is to execute

    git clone --recursive https://github.com/leinkemmer/Ensign.git

To build the software the following are needed
- CMake
- C++14 compatible C++ compiler
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

