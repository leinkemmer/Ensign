#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <generic/storage.hpp>

TEST_CASE( "multi_array basic operations", "[multi_array]" ) {

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
        REQUIRE( a(1,2) == Approx(3.5) );
    }

    # ifdef __CUDACC__
    SECTION("TEST GPU"){
        multi_array<double,2> a_cpu({2,3});
        // define on cpu
        //
        multi_array<double,2> a_gpu({2,3},stloc::device);
        a_gpu = a_cpu;


        // computation on gpu
        
        a_cpu = a_gpu;
        cout << a_gpu.data() << endl;
        cout << a_cpu(0,0) << endl;
    }
  #endif
}
