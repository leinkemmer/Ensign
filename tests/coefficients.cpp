#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <generic/storage.hpp>
#include <lr/coefficients.hpp>

TEST_CASE( "coefficients", "[coefficients]" ) {
    multi_array<double,2> a({10,11}), b({10,11}), out({10,11});

    coeff(a, b, 0.3, out);
}

