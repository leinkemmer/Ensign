#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <generic/storage.hpp>

#ifdef __CUDACC__
  cublasHandle_t  handle;
#endif

using namespace Ensign;


TEST_CASE( "multi_array basic operations", "[multi_array]" ) {

  #ifdef __CUDACC__
    cublasCreate(&handle);
  #endif

    multi_array<double, 2> a({10,11});

    SECTION("comparison"){
      multi_array<double, 2> aa({2,3});
      multi_array<double, 2> bb({2,3});
      multi_array<double, 2> cc({2,3});
      multi_array<double, 2> dd({2,2});

      for(int i = 0; i< 2; i++){
        for(int j = 0; j<3; j++){
          aa(i,j) = i + 2*j;
          bb(i,j) = i + 2*j;
          cc(i,j) = j + 2*i;
        }
      }
      REQUIRE( (aa == bb) == true);
      REQUIRE( (aa == cc) == false);
      REQUIRE( (aa == dd) == false);
    }

    SECTION("shape") {
        REQUIRE( a.shape() == array<Index,2>({10, 11}) );
        REQUIRE( a.num_elements() == 10*11 );
    }

    SECTION("accessing elements") {
        a(1,2) = 3.5;
        REQUIRE( a(1,2) == Catch::Approx(3.5) );
    }

    # ifdef __CUDACC__
    SECTION("TEST GPU"){
        multi_array<double,2> a_cpu1({1,2});
        multi_array<double,2> a_cpu2({1,2});
        multi_array<double,2> a_gpu({1,2},stloc::device);

        a_cpu1(0,0) = 2.2;
        a_cpu1(0,1) = 1.5;

        a_gpu = a_cpu1;
        a_cpu2 = a_cpu1;

        a_cpu1 = a_gpu;

        REQUIRE( a_cpu1(0,0) == Catch::Approx(2.2) );
        REQUIRE( a_cpu1(0,1) == Catch::Approx(1.5) );
        REQUIRE( bool(a_cpu1 == a_cpu2));
    }

    SECTION("PTW OPERATIONS"){
      multi_array<double,1> a_cpu({10});
      multi_array<double,1> a({10},stloc::device);
      multi_array<double,1> a2_cpu({10});

      for(int i = 0; i < 10; i++){
        a_cpu(i) = i;
      }
      a = a_cpu;

      a += 2.0;
      a2_cpu = a;

      a_cpu += 2.0;

      cout << a_cpu << endl;
      cout << a2_cpu << endl;

    }

  #endif

  #ifdef __CUDACC__
    cublasDestroy(handle);
  #endif
}
