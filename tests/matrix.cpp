#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <generic/matrix.hpp>

TEST_CASE( "matrix basic operations", "[matrix]" ) {

    multi_array<double, 2> a({10,11});
    set_zero(a);
    
    multi_array<float, 2> b({10,11});
    set_zero(b);

    //SECTION("shape") {
    //    REQUIRE( a.shape() == array<Index,2>({10, 11}) );
    //    REQUIRE( a.num_elements() == 10*11 );
    //}
    //
    //SECTION("accessing elements") {
    //    a(1,2) = 3.5;
    //    REQUIRE( a(1,2) == Approx(3.5) );
    //}
}

