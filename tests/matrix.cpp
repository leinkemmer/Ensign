#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <generic/matrix.hpp>


TEST_CASE( "matrix basic operations", "[matrix]" ) {

  blas_ops blas;

  SECTION("initializations (CPU)"){
    multi_array<double, 2> a({2,3});
    set_zero(a);

    multi_array<float, 2> b({2,3});
    set_zero(b);

    multi_array<float, 2> zzf({2,3});
    multi_array<double, 2> zzd({2,3});

    for(int i = 0; i < 2; i++){
      for(int j = 0; j < 3; j++){
        zzd(i,j) = 0.0;
        zzf(i,j) = 0.0;
      }
    }

    REQUIRE (bool(a == zzd));
    REQUIRE (bool(b == zzf));

    multi_array<double,2> c({3,3});
    set_identity(c);

    multi_array<float,2> d({3,3});
    set_identity(d);

    multi_array<float, 2> iif({3,3});
    multi_array<double, 2> iid({3,3});
    set_zero(iif);
    set_zero(iid);

    for(int i = 0; i < 3; i++){
      iif(i,i) = 1.0;
      iid(i,i) = 1.0;
    }

    REQUIRE(bool(iif == d));
    REQUIRE(bool(iid == c));
  }

  #ifdef __CUDACC__
  SECTION("initializations (GPU)"){
    multi_array<double, 2> a({2,3},stloc::device);
    set_zero(a);

    multi_array<float, 2> b({2,3},stloc::device);
    set_zero(b);

    multi_array<float, 2> zzf({2,3});
    multi_array<double, 2> zzd({2,3});

    for(int i = 0; i < 2; i++){
      for(int j = 0; j < 3; j++){
        zzd(i,j) = 0.0;
        zzf(i,j) = 0.0;
      }
    }
    multi_array<double,2> _a(a.shape());
    _a = a;
    multi_array<float,2> _b(b.shape());
    _b = b;

    REQUIRE (bool(_a == zzd));
    REQUIRE (bool(_b == zzf));

    multi_array<double,2> c({3,3},stloc::device);
    set_identity(c);

    multi_array<float,2> d({3,3},stloc::device);
    set_identity(d);

    multi_array<float, 2> iif({3,3});
    multi_array<double, 2> iid({3,3});
    set_zero(iif);
    set_zero(iid);

    for(int i = 0; i < 3; i++){
      iif(i,i) = 1.0;
      iid(i,i) = 1.0;
    }

    multi_array<double,2> _c(c.shape());
    _c = c;
    multi_array<float,2> _d(d.shape());
    _d = d;

    REQUIRE(bool(iif == _d));
    REQUIRE(bool(iid == _c));

  }
  #endif

  SECTION("Pointwise multiplication"){
    multi_array<double,2> A({3,2});
    multi_array<double,2> R({3,2});
    multi_array<double,1> w({3});
    w(0) = 2.0; w(1) = 3.0; w(2) = 4.0;

    multi_array<float,2> Af({3,2});
    multi_array<float,2> Rf({3,2});
    multi_array<float,1> wf({3});
    wf(0) = 2.0; wf(1) = 3.0; wf(2) = 4.0;


    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 2; j++){
        A(i,j) = 1.0;
        R(i,j) = w(i);
        Af(i,j) = 1.0;
        Rf(i,j) = wf(i);
      }
    }

    ptw_mult_row(A,w,A);
    ptw_mult_row(Af,wf,Af);

    REQUIRE(bool(A==R));
    REQUIRE(bool(Af==Rf));

  }

  SECTION("A*B multiplication"){
    multi_array<double,2> A({2,3});
    multi_array<double,2> B({3,4});
    multi_array<double,2> AB({2,4});
    multi_array<double,2> RAB({2,4});

    multi_array<float,2> Af({2,3});
    multi_array<float,2> Bf({3,4});
    multi_array<float,2> ABf({2,4});
    multi_array<float,2> RABf({2,4});


    for(int i = 0; i<2; i++){
      for(int j = 0; j<3; j++){
        A(i,j) = i + 2*j;
        Af(i,j) = i + 2*j;
      }
    }

    for(int i = 0; i<3; i++){
      for(int j = 0; j<4; j++){
        B(i,j) = i + 3*j;
        Bf(i,j) = i + 3*j;
      }
    }

    set_zero(RAB);
    set_zero(RABf);
    for(int i = 0; i<2; i++){
      for(int j = 0; j<4; j++){
        for(int k = 0; k< 3; k++){
          RAB(i,j) += A(i,k)*B(k,j);
          RABf(i,j) += Af(i,k)*Bf(k,j);
        }
      }
    }

    blas.matmul(A,B,AB);
    REQUIRE(bool(RAB == AB));
    blas.matmul(Af,Bf,ABf);
    REQUIRE(bool(RABf == ABf));

  }

  #ifdef __CUDACC__
  SECTION("A*B multiplication on GPU"){
    multi_array<double,2> A({2,3},stloc::device);
    multi_array<double,2> B({3,4},stloc::device);
    multi_array<double,2> AB({2,4},stloc::device);
    multi_array<double,2> _A({2,3});
    multi_array<double,2> _B({3,4});
    multi_array<double,2> _AB({2,4});
    multi_array<double,2> _RAB({2,4});

    multi_array<float,2> Af({2,3},stloc::device);
    multi_array<float,2> Bf({3,4},stloc::device);
    multi_array<float,2> ABf({2,4},stloc::device);
    multi_array<float,2> _Af({2,3});
    multi_array<float,2> _Bf({3,4});
    multi_array<float,2> _ABf({2,4});
    multi_array<float,2> _RABf({2,4});


    for(int i = 0; i<2; i++){
      for(int j = 0; j<3; j++){
        _A(i,j) = i + 2*j;
        _Af(i,j) = i + 2*j;
      }
    }
    A = _A;
    Af = _Af;

    for(int i = 0; i<3; i++){
      for(int j = 0; j<4; j++){
        _B(i,j) = i + 3*j;
        _Bf(i,j) = i + 3*j;
      }
    }
    B = _B;
    Bf = _Bf;

    blas.matmul(_A,_B,_RAB);
    blas.matmul(A,B,AB);
    _AB = AB;
    REQUIRE(bool(_RAB == _AB));

    blas.matmul(_Af,_Bf,_RABf);
    blas.matmul(Af,Bf,ABf);
    _ABf = ABf;
    REQUIRE(bool(_RABf == _ABf));
  }
  #endif

  SECTION("At*B multiplication"){
    multi_array<double,2> A({3,2});
    multi_array<double,2> B({3,4});
    multi_array<double,2> AtB({2,4});
    multi_array<double,2> RAtB({2,4});
    multi_array<float,2> Af({3,2});
    multi_array<float,2> Bf({3,4});
    multi_array<float,2> AtBf({2,4});
    multi_array<float,2> RAtBf({2,4});


    for(int i = 0; i<3; i++){
      for(int j = 0; j<2; j++){
        A(i,j) = j + 2*i;
        Af(i,j) = j + 2*i;
      }
    }

    for(int i = 0; i<3; i++){
      for(int j = 0; j<4; j++){
        B(i,j) = i + 3*j;
        Bf(i,j) = i + 3*j;
      }
    }

    set_zero(RAtB);
    set_zero(RAtBf);
    for(int i = 0; i<2; i++){
      for(int j = 0; j<4; j++){
        for(int k = 0; k< 3; k++){
          RAtB(i,j) += A(k,i)*B(k,j);
          RAtBf(i,j) += Af(k,i)*Bf(k,j);
        }
      }
    }

    blas.matmul_transa(A,B,AtB);
    REQUIRE(bool(RAtB == AtB));
    blas.matmul_transa(Af,Bf,AtBf);
    REQUIRE(bool(RAtBf == AtBf));
  }

  #ifdef __CUDACC__
  SECTION("At*B multiplication on GPU"){
    multi_array<double,2> A({3,2},stloc::device);
    multi_array<double,2> B({3,4},stloc::device);
    multi_array<double,2> AtB({2,4},stloc::device);
    multi_array<double,2> _A({3,2});
    multi_array<double,2> _B({3,4});
    multi_array<double,2> _AtB({2,4});
    multi_array<double,2> _RAtB({2,4});

    multi_array<float,2> Af({3,2},stloc::device);
    multi_array<float,2> Bf({3,4},stloc::device);
    multi_array<float,2> AtBf({2,4},stloc::device);
    multi_array<float,2> _Af({3,2});
    multi_array<float,2> _Bf({3,4});
    multi_array<float,2> _AtBf({2,4});
    multi_array<float,2> _RAtBf({2,4});

    for(int i = 0; i<3; i++){
      for(int j = 0; j<2; j++){
        _A(i,j) = j + 2*i;
        _Af(i,j) = j + 2*i;
      }
    }
    A = _A;
    Af = _Af;

    for(int i = 0; i<3; i++){
      for(int j = 0; j<4; j++){
        _B(i,j) = i + 3*j;
        _Bf(i,j) = i + 3*j;
      }
    }
    B = _B;
    Bf = _Bf;

    blas.matmul_transa(A,B,AtB);
    blas.matmul_transa(_A,_B,_RAtB);
    _AtB = AtB;
    REQUIRE(bool(_AtB == _RAtB));

    blas.matmul_transa(Af,Bf,AtBf);
    blas.matmul_transa(_Af,_Bf,_RAtBf);
    _AtBf = AtBf;
    REQUIRE(bool(_AtBf == _RAtBf));
  }
  #endif

  SECTION("A*Bt multiplication"){
    multi_array<double,2> A({2,3});
    multi_array<double,2> B({4,3});
    multi_array<double,2> ABt({2,4});
    multi_array<double,2> RABt({2,4});
    multi_array<float,2> Af({2,3});
    multi_array<float,2> Bf({4,3});
    multi_array<float,2> ABtf({2,4});
    multi_array<float,2> RABtf({2,4});

    for(int i = 0; i<2; i++){
      for(int j = 0; j<3; j++){
        A(i,j) = i + 2*j;
        Af(i,j) = i + 2*j;
      }
    }
    for(int i = 0; i<4; i++){
      for(int j = 0; j<3; j++){
        B(i,j) = j + 3*i;
        Bf(i,j) = j + 3*i;
      }
    }

    set_zero(RABt);
    set_zero(RABtf);
    for(int i = 0; i<2; i++){
      for(int j = 0; j<4; j++){
        for(int k = 0; k< 3; k++){
          RABt(i,j) += A(i,k)*B(j,k);
          RABtf(i,j) += Af(i,k)*Bf(j,k);
        }
      }
    }

    blas.matmul_transb(A,B,ABt);
    REQUIRE(bool(RABt == ABt));
    blas.matmul_transb(Af,Bf,ABtf);
    REQUIRE(bool(RABtf == ABtf));
  }

  #ifdef __CUDACC__
  SECTION("A*Bt multiplication on GPU"){
    multi_array<double,2> A({2,3},stloc::device);
    multi_array<double,2> B({4,3},stloc::device);
    multi_array<double,2> ABt({2,4},stloc::device);
    multi_array<double,2> _A({2,3});
    multi_array<double,2> _B({4,3});
    multi_array<double,2> _ABt({2,4});
    multi_array<double,2> _RABt({2,4});

    multi_array<float,2> Af({2,3},stloc::device);
    multi_array<float,2> Bf({4,3},stloc::device);
    multi_array<float,2> ABtf({2,4},stloc::device);
    multi_array<float,2> _Af({2,3});
    multi_array<float,2> _Bf({4,3});
    multi_array<float,2> _ABtf({2,4});
    multi_array<float,2> _RABtf({2,4});

    for(int i = 0; i<2; i++){
      for(int j = 0; j<3; j++){
        _A(i,j) = i + 2*j;
        _Af(i,j) = i + 2*j;
      }
    }
    A = _A;
    Af = _Af;

    for(int i = 0; i<4; i++){
      for(int j = 0; j<3; j++){
        _B(i,j) = j + 3*i;
        _Bf(i,j) = j + 3*i;
      }
    }
    B = _B;
    Bf = _Bf;

    blas.matmul_transb(_A,_B,_RABt);
    blas.matmul_transb(A,B,ABt);
    _ABt = ABt;
    REQUIRE(bool(_RABt == _ABt));

    blas.matmul_transb(_Af,_Bf,_RABtf);
    blas.matmul_transb(Af,Bf,ABtf);
    _ABtf = ABtf;
    REQUIRE(bool(_RABtf == _ABtf));
  }
  #endif

  SECTION("At*Bt multiplication"){
    multi_array<double,2> A({3,2});
    multi_array<double,2> B({4,3});
    multi_array<double,2> AtBt({2,4});
    multi_array<double,2> RAtBt({2,4});
    multi_array<float,2> Af({3,2});
    multi_array<float,2> Bf({4,3});
    multi_array<float,2> AtBtf({2,4});
    multi_array<float,2> RAtBtf({2,4});

    for(int i = 0; i<3; i++){
      for(int j = 0; j<2; j++){
        A(i,j) = j + 2*i;
        Af(i,j) = j + 2*i;
      }
    }
    for(int i = 0; i<4; i++){
      for(int j = 0; j<3; j++){
        B(i,j) = j + 3*i;
        Bf(i,j) = j + 3*i;
      }
    }

    set_zero(RAtBt);
    set_zero(RAtBtf);
    for(int i = 0; i<2; i++){
      for(int j = 0; j<4; j++){
        for(int k = 0; k< 3; k++){
          RAtBt(i,j) += A(k,i)*B(j,k);
          RAtBtf(i,j) += Af(k,i)*Bf(j,k);
        }
      }
    }

    blas.matmul_transab(A,B,AtBt);
    REQUIRE(bool(RAtBt == AtBt));
    blas.matmul_transab(Af,Bf,AtBtf);
    REQUIRE(bool(RAtBtf == AtBtf));

  }

  #ifdef __CUDACC__
  SECTION("At*Bt multiplication on GPU"){
    multi_array<double,2> A({3,2},stloc::device);
    multi_array<double,2> B({4,3},stloc::device);
    multi_array<double,2> AtBt({2,4},stloc::device);
    multi_array<double,2> _A({3,2});
    multi_array<double,2> _B({4,3});
    multi_array<double,2> _AtBt({2,4});
    multi_array<double,2> _RAtBt({2,4});

    multi_array<float,2> Af({3,2},stloc::device);
    multi_array<float,2> Bf({4,3},stloc::device);
    multi_array<float,2> AtBtf({2,4},stloc::device);
    multi_array<float,2> _Af({3,2});
    multi_array<float,2> _Bf({4,3});
    multi_array<float,2> _AtBtf({2,4});
    multi_array<float,2> _RAtBtf({2,4});

    for(int i = 0; i<3; i++){
      for(int j = 0; j<2; j++){
        _A(i,j) = j + 2*i;
        _Af(i,j) = j + 2*i;
      }
    }
    A = _A;
    Af = _Af;

    for(int i = 0; i<4; i++){
      for(int j = 0; j<3; j++){
        _B(i,j) = j + 3*i;
        _Bf(i,j) = j + 3*i;
      }
    }
    B = _B;
    Bf = _Bf;

    blas.matmul_transab(A,B,AtBt);
    blas.matmul_transab(_A,_B,_RAtBt);
    _AtBt = AtBt;
    REQUIRE(bool(_RAtBt == _AtBt));

    blas.matmul_transab(Af,Bf,AtBtf);
    blas.matmul_transab(_Af,_Bf,_RAtBtf);
    _AtBtf = AtBtf;
    REQUIRE(bool(_RAtBtf == _AtBtf));
  }
  #endif


    SECTION("Complex multiplication on CPU"){
      multi_array<complex<double>,2> A({2,3});
      multi_array<complex<double>,2> B({3,4});
      multi_array<complex<double>,2> AB({2,4});
      multi_array<complex<double>,2> RAB({2,4});

      complex<double> one(0.0,1.0);

      for(int i = 0; i<2; i++){
        for(int j = 0; j<3; j++){
          A(i,j) = complex<double>(i + 2*j) + one;
          cout << A(i,j) << endl;
        }
      }

      for(int i = 0; i<3; i++){
        for(int j = 0; j<4; j++){
          B(i,j) = complex<double>(i + 3*j) + one;
          cout << B(i,j) << endl;
        }
      }

      set_zero(RAB);
      for(int i = 0; i<2; i++){
        for(int j = 0; j<4; j++){
          for(int k = 0; k< 3; k++){
            RAB(i,j) += A(i,k)*B(k,j);
          }
        }
      }

      blas.matmul(A,B,AB);

      for(int i = 0; i<2; i++){
        for(int j = 0; j<4; j++){
          cout << AB(i,j) << endl;
        }
      }

  //    REQUIRE(bool(RAB == AB)); TO DO

    }

  #ifdef __CUDACC__
  SECTION("Matvec (CPU and GPU)"){

    multi_array<double,2> A({2,3},stloc::device);
    multi_array<double,1> b({3},stloc::device);
    multi_array<double,1> Ab({2},stloc::device);
    multi_array<double,2> _A({2,3});
    multi_array<double,1> _b({3});
    multi_array<double,1> _Ab2({2});
    multi_array<double,1> _Ab({2});
    multi_array<double,1> _RAb({2});

    for(int i = 0; i<2; i++){
      for(int j = 0; j<3; j++){
        _A(i,j) = i + 2*j;
      }
    }
    A = _A;

    for(int i = 0; i<3; i++){
      _b(i) = 1+i;
      }
    b = _b;

    _RAb(0) = 0.0;
    _RAb(1) = 0.0;

    for(int i = 0; i<2; i++){
        for(int k = 0; k< 3; k++){
          _RAb(i) += _A(i,k)*_b(k);
        }
    }

    blas.matvec(_A,_b,_Ab2);
    blas.matvec(A,b,Ab);
    _Ab = Ab;
    REQUIRE(bool(_RAb == _Ab));
    REQUIRE(bool(_RAb == _Ab2));

  }
  #endif

  #ifdef __CUDACC__
  SECTION("Matvectransp (CPU and GPU)"){

    multi_array<double,2> A({3,2},stloc::device);
    multi_array<double,1> b({3},stloc::device);
    multi_array<double,1> Ab({2},stloc::device);
    multi_array<double,2> _A({3,2});
    multi_array<double,1> _b({3});
    multi_array<double,1> _Ab2({2});
    multi_array<double,1> _Ab({2});
    multi_array<double,1> _RAb({2});

    for(int i = 0; i<3; i++){
      for(int j = 0; j<2; j++){
        _A(i,j) = i + 2*j;
      }
    }
    A = _A;

    for(int i = 0; i<3; i++){
      _b(i) = 1+i;
      }
    b = _b;

    for(int i = 0; i<3; i++){
        for(int k = 0; k< 2; k++){
          _RAb(k) += _A(i,k)*_b(i);
        }
    }

    cout << _A << endl;
    cout << _b << endl;

    blas.matvec_trans(_A,_b,_Ab2);
    blas.matvec_trans(A,b,Ab);
    _Ab = Ab;

    REQUIRE(bool(_RAb == _Ab));
    REQUIRE(bool(_RAb == _Ab2));

  }
  #endif

  #ifdef __CUDACC__
  SECTION("Fill const vector in GPU"){
    multi_array<double,1> v({20},stloc::device);
    multi_array<double,1> v_cpu({20});

    double alpha = 2.2;
    set_const(v,alpha);
    v_cpu = v;
    cout << v_cpu << endl;

  }
  #endif

}
