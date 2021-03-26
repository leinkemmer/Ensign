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
        multi_array<double,2> a_cpu1({1,2});
        multi_array<double,2> a_cpu2({1,2});

        a_cpu1(0,0) = 2.2;
        a_cpu1(0,1) = 1.5;
        cout << "Should allocate on GPU" << endl;
        multi_array<double,2> a_gpu({1,2},stloc::device);
        cout << "Finish Should allocate on GPU" << endl;
        cout << "Transfer CPU->GPU" << endl;
        a_gpu.transfer(a_cpu1);
        cout << "Finish Transfer CPU->GPU" << endl;

        cout << "Transfer GPU->CPU wasting memory" << endl;
        a_cpu1 = a_gpu;
        cout << "Finish Transfer GPU->CPU wasting memory" << endl;

        cout << "Transfer GPU/CPU without wasting memory" << endl;
        a_cpu2.transfer(a_gpu);
        cout << "Finish Transfer GPU/CPU without wasting memory" << endl;


        REQUIRE( a_cpu1(0,0) == Approx(2.2) );
        REQUIRE( a_cpu1(0,1) == Approx(1.5) );
        REQUIRE( bool(a_cpu1 == a_cpu2));
      
    }
  #endif
}
