#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <generic/storage.hpp>

TEST_CASE( "multi_array basic operations", "[multi_array]" ) {

    multi_array<double, 2> a({10,11});

    SECTION("shape") {
        REQUIRE( a.shape() == array<Index,2>({10, 11}) );
        REQUIRE( a.num_elements() == 10*11 );
    }
    
    SECTION("accessing elements") {
        a(1,2) = 3.5;
        REQUIRE( a(1,2) == Approx(3.5) );
    }
}

