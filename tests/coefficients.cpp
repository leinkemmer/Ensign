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

    coeff_mat(a,b,co,out2);

    multi_array<double,2> R({3,3});
    matmul_transa(a,b,R);

    R = R * co;

    REQUIRE((bool)(R == out));
    REQUIRE((bool)(R == out2));

    multi_array<double,2> aa({3,2}), bb({3,2}), outout({2,2});
    set_zero(aa);
    aa(0,0) = 3.0 / 5.0;
    aa(2,0) = 4.0 / 5.0;
    aa(1,1) = 1.0;

    bb = aa;
    coeff(aa, bb, 1.0, outout);

    multi_array<double,2> RR({2,2});
    set_identity(RR);

    REQUIRE((bool)(RR == outout));

    array<double,3> ww = {-1.0,2.0};
    multi_array<double,2> out3({3,3});

    coeff_gen(a, b, ww.begin(), out3);

    multi_array<double,2> R3({3,3});

    R3(0,0) = 4.0; R3(0,1) = 8.0; R3(0,2) = 12.0;
    R3(1,0) = 10.0; R3(1,1) = 18.0; R3(1,2) = 26.0;
    R3(2,0) = 16.0; R3(2,1) = 28.0; R3(2,2) = 40.0;

    REQUIRE((bool)(R3 == out3));

}
;
