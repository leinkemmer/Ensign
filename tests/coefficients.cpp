#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <generic/matrix.hpp>
#include <generic/storage.hpp>

#include <lr/coefficients.hpp>


TEST_CASE( "coefficients", "[coefficients]" ) {
    multi_array<double,2> a({2,3}), b({2,3}), out({3,3}), out2({3,3});

    for(int i = 0; i<2; i++){
      for(int j = 0; j<3; j++){
        a(i,j) = i + 2*j;
        b(i,j) = 1+i + 2*j;
      }
    }

    double co = 0.3;

    coeff(a, b, co, out);

    Index N = a.shape()[0];
    int r_a = a.shape()[1];
    int r_b = b.shape()[1];

    double* ptr_j;
    double* ptr_l;
    set_zero(out2);
    for(int j = 0; j < r_a; j++){
      for(int l = 0; l < r_b; l++){
        ptr_j = a.extract({j});
        ptr_l = b.extract({l});
        for(int i = 0; i < N; i++){
          out2(j,l) += ptr_j[i]*ptr_l[i]*co;
        }
      }
    }

    REQUIRE(bool(out == out2));

    multi_array<double,2> aa({3,2}), bb({3,2}), outout({2,2});
    set_zero(aa);
    aa(0,0) = 3.0 / 5.0;
    aa(2,0) = 4.0 / 5.0;
    aa(1,1) = 1.0;

    bb = aa;
    coeff(aa, bb, 1.0, outout);

    multi_array<double,2> RR({2,2});
    set_identity(RR);

    REQUIRE(bool(RR == outout));

    multi_array<double,2> aaa({3,2}), bbb({3,2}), out3({2,2});

    for(int i = 0; i<3; i++){
      for(int j = 0; j<2; j++){
        aaa(i,j) = i + 3*j;
        bbb(i,j) = 1+i + 3*j;
      }
    }

    array<double,3> ww = {-1.0,2.0,5.0};

    coeff(aaa, bbb, ww.begin(), out3);

    multi_array<double,2> R3({2,2});

    R3(0,0) = 34.0; R3(0,1) = 70.0;
    R3(1,0) = 88.0; R3(1,1) = 178.0;

    REQUIRE(bool(R3 == out3));

}
;
