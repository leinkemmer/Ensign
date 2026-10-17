# Installation

## Requirements

- CMake
- A C++11 compatible C++ compiler (OpenMP support recommended)
- A Fortran compiler (only if the bundled OpenBLAS is used)
- CUDA (only if GPU support is desired)

On Ubuntu:

    sudo apt install build-essential gfortran cmake git

## Building

    git clone https://github.com/leinkemmer/Ensign.git
    cd Ensign
    mkdir build
    cd build
    cmake ..
    make

This builds the library, the examples and the tests. The executables of the examples
end up in `build/examples`.

### CMake options

| Option | Default | Meaning |
|---|---|---|
| `OPENMP` | `ON` | Enable OpenMP parallelization |
| `CUDA_ENABLED` | `OFF` | Enable GPU support (requires CUDA) |
| `MKL_ENABLED` | `OFF` | Use Intel MKL instead of OpenBLAS |
| `Sanitizer` | `OFF` | Enable the address sanitizer |
| `CMAKE_BUILD_TYPE` | | Use `Release` for production runs |

Example with GPU support:

    cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=ON ..

### Intel MKL

    export MKLROOT=/opt/intel/mkl
    cmake -DMKL_ENABLED=ON -DCUDA_ENABLED=ON ..

## macOS

### OpenBLAS
If OpenBLAS is used as a BLAS backend, a Fortran compiler has to be installed. Apple Clang does
not ship one, so install `gcc-14` via brew, which provides `gfortran-14`.
If CMake does not find the Fortran compiler, set

    export FC=/path/to/fortran/compiler

and possibly the CMake cache entry `CMAKE_Fortran_COMPILER` to its full path.

### OpenMP
Apple Clang does not officially support OpenMP. Use GCC instead:

    brew install gcc@14
    brew info gcc@14   # find the installation path

and set

    export CC=/path/to/gcc-14
    export CXX=/path/to/g++-14
    export FC=/path/to/gfortran-14

before running CMake.

## Using Ensign in your own project

Ensign exports a CMake target `Ensign::Ensign`:

```cmake
add_executable(my_sim my_sim.cpp)
target_link_libraries(my_sim Ensign::Ensign)
```

## Building this documentation

    sudo apt install doxygen
    docs/serve.sh         # builds the API reference and serves http://127.0.0.1:8000
    docs/serve.sh build   # only write the static site to site/

The script creates a Python virtual environment in `~/.cache/ensign-docs-venv` (override with `DOCS_VENV`).
