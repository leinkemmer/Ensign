#pragma once

#include <generic/common.hpp>
#include <generic/utility.hpp>
#include <generic/kernels.hpp>

template<class T, size_t d>
struct multi_array {
  array<Index,d> e;
  T* v;
  stloc sl;

  multi_array(stloc _sl=stloc::host) : sl(_sl), v(nullptr) {
    fill(e.begin(), e.end(), 0);
  }

  multi_array(array<Index,d> _e, stloc _sl=stloc::host) : sl(_sl) {
    resize(_e);
  }

  // copy constructor
  multi_array(const multi_array& ma) {
    sl = ma.sl;
    resize(ma.e);
    if(sl == stloc::host) {
      std::copy(ma.data(), ma.data()+ma.num_elements(), v);
    } else {
      #ifdef __CUDACC__
      cudaMemcpy(v, ma.data(), sizeof(T)*ma.num_elements(),
      cudaMemcpyDeviceToDevice);
      #else
      cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
      #endif
    }
  }

  // copy and swap assignment operator
  multi_array& operator=(const multi_array& ma) {
    if(v == nullptr)
    resize(ma.e);
    else if(e != ma.e) {
      cout << "ERROR: array dimensions do not match in assignment" << endl;
      exit(1);
    }

    if(ma.sl == stloc::host && sl == stloc::host) { // both on CPU
      std::copy(ma.data(), ma.data()+ma.num_elements(), v);
    } else {
      #ifdef __CUDACC__
      if(sl == stloc::host){ // dst on CPU
        cudaMemcpy(v, ma.data(), sizeof(T)*ma.num_elements(),
        cudaMemcpyDeviceToHost);
      } else if(ma.sl == stloc::host){ // src on CPU
        cudaMemcpy(v, ma.data(), sizeof(T)*ma.num_elements(),
        cudaMemcpyHostToDevice);
      } else {             // both src and dst on GPU
        cudaMemcpy(v, ma.data(), sizeof(T)*ma.num_elements(),
        cudaMemcpyDeviceToDevice);
      }
      #else
      cout << "ERROR: compiled without GPU support" << endl;
      exit(1);
      #endif
    }

    return *this;
  }

  void resize(array<Index,d> _e) {
    e = _e;
    Index num_elements = prod(e);
    if(sl == stloc::host) {
      v = (T*)aligned_alloc(64,sizeof(T)*num_elements);
    } else {
      #ifdef __CUDACC__
      v = (T*)gpu_malloc(sizeof(T)*num_elements);
      #else
      cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
      #endif
    }
  }

  ~multi_array() {
    if(v != nullptr) {
      if(sl == stloc::host) {
        free(v);
      } else {
        #ifdef __CUDACC__
        cudaFree(v);
        #else
        cout << "ERROR: compiled without GPU support" << __FILE__ << ":"
        << __LINE__ << endl;
        exit(1);
        #endif
      }
    }
  }

  Index linear_idx(array<Index,d> idx) const {
    Index k=0;
    Index stride = 1;
    for(size_t i=0;i<d;i++) {
      k += stride*idx[i];
      stride *= e[i];
    }
    assert(k < num_elements());
    return k;
  }

  void to_cplx(multi_array<complex<T>,d>& out) {
    std::transform(begin(), end(), out.begin(), [](const T& c){return complex<T>(c,T(0.0));} );
  }

  void save_vector(std::string fn) {
    std::ofstream fs(fn.c_str(), std::ios::binary);
    fs.write((char*)data(), sizeof(T)*num_elements());
  }


  T& operator()(array<Index,d> idx) {
    return v[linear_idx(idx)];
  }

  // TODO: if called as (z,0) this gives a -Wnarrowing warning. These warnings
  // are turned off in the build system at the moment.
  template<typename... Ints>
  T& operator()(Ints&&... idx) {
    static_assert(sizeof...(Ints) == d, "wrong number of arguments to ().");
    return v[linear_idx(array<Index,d>({idx...}))];
  }

  T* extract(array<Index,d-1> idx_r) {
    array<Index,d> idx;
    std::copy(std::begin(idx_r), std::end(idx_r), std::begin(idx)+1);
    idx[0] = 0;
    return &v[linear_idx(idx)];
  }

  const T* extract(array<Index,d-1> idx_r) const {
    array<Index,d> idx;
    std::copy(std::begin(idx_r), std::end(idx_r), std::begin(idx)+1);
    idx[0] = 0;
    return &v[linear_idx(idx)];
  }


  array<Index,d> shape() const {
    return e;
  }

  T* data() const {
    return &v[0];
  }

  Index num_elements() const {
    return prod(e);
  }

  T* begin() const noexcept {
    return &v[0];
  }

  T* end() const noexcept {
    return &v[num_elements()];
  }

  multi_array& operator+=(const multi_array& lhs) {
    /*
    if((sl == stloc::host) && (lhs.sl == stloc::host)){
      std::transform(begin(), end(), lhs.begin(), begin(), [](T& a, T& b){return a+b;} );
    }else if ((sl == stloc::device) && (lhs.sl == stloc::device)){
      #ifdef __CUDACC__
        ptw_sum<<<(num_elements()+n_threads-1)/n_threads,n_threads>>>(num_elements(),begin(),lhs.begin());
      #endif
    }else{
      cout << "ERROR: either both on CPU or on GPU" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
    }
    */
    std::transform(begin(), end(), lhs.begin(), begin(), [](T& a, T& b){return a+b;} );
    return *this;
  }

  multi_array& operator+=(const T scalar) {
    if(sl == stloc::host){
      std::transform(begin(), end(), begin(), [&scalar](T& a){return scalar+a;} );
    }else{
      #ifdef __CUDACC__
        ptw_sum_scal<<<(num_elements()+n_threads-1)/n_threads,n_threads>>>(num_elements(),begin(),scalar);
      #endif
    }
    return *this;
  }

  multi_array& operator-=(const multi_array& lhs) {
    /*
    if((sl == stloc::host) && (lhs.sl == stloc::host)){
      std::transform(begin(), end(), lhs.begin(), begin(), [](T& a, T& b){return a-b;} );
    }else if ((sl == stloc::device) && (lhs.sl == stloc::device)){
      #ifdef __CUDACC__
        ptw_diff<<<(num_elements()+n_threads-1)/n_threads,n_threads>>>(num_elements(),begin(),lhs.begin());
      #endif
    }else{
      cout << "ERROR: either both on CPU or on GPU" << __FILE__ << ":"
      << __LINE__ << endl;
      exit(1);
    }
    */
    std::transform(begin(), end(), lhs.begin(), begin(), [](T& a, T& b){return a-b;} );
    return *this;
  }

  multi_array& operator*=(const T scalar) {
    if(sl == stloc::host){
      std::transform(begin(), end(), begin(), [&scalar](T& a){return scalar*a;} );
    }else{
      #ifdef __CUDACC__
        //ptw_mult_scal<<<2,2>>>(num_elements(),begin(),scalar); TO BE FIXED FOR COMPLEX NUMBERS
      #endif
    }
    return *this;
  }

  multi_array& operator/=(const T scalar) {
    std::transform(begin(), end(), begin(), [&scalar](T& a){return a/scalar;} );
    return *this;
  }

  multi_array operator+(const multi_array& lhs) {
    multi_array<T,d> out(e);
    std::transform(begin(), end(), lhs.begin(), out.begin(), [](T& a, T& b){return a+b;} );
    return out;
  }

  multi_array operator-(const multi_array& lhs) {
    multi_array<T,d> out(e);
    std::transform(begin(), end(), lhs.begin(), out.begin(), [](T& a, T& b){return a-b;} );
    return out;
  }

  multi_array operator*(const T scalar) {
    multi_array<T,d> out(e);
    std::transform(begin(), end(), out.begin(), [&scalar](const T& c){return c*scalar;} );
    return out;
  }

  friend std::ostream &operator<<(std::ostream &os, const multi_array &A) {

    if(A.shape().size() == 1){
      for (Index i=0; i < A.shape()[0]; i++) {
        os << A.v[i] << '\n';
      }
    }else if(A.shape().size() == 2){
      for (Index i=0; i < A.shape()[0]; i++) {
        for (Index j=0; j < A.shape()[1]; j++) {
          os << A.v[i + j*A.shape()[0]] << "  " ;
        }
        os << '\n';
      }
    } else {
      cout << "ERROR: dimension for cout not yet supported" << endl;
      exit(1);
    }
    return os;
  }


  bool operator==(const multi_array& lhs){
    if (lhs.shape() != shape()){
      return false;
    } else {
      for(Index i=0;i<lhs.num_elements();i++){
        if(std::abs((lhs.v[i] - v[i])) > T(1000)*std::numeric_limits<T>::epsilon())
        return false;
      }
      return true;
    }
  }
};
