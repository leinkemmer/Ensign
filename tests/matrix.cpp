#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <generic/matrix.hpp>

TEST_CASE( "matrix basic operations", "[matrix]" ) {

  SECTION("initializations"){
    multi_array<double, 2> a({2,3});
    set_zero(a);

    multi_array<float, 2> b({2,3});
    set_zero(b);

    for(int i = 0; i < 2; i++){
      for(int j = 0; j < 3; j++){
        REQUIRE(a(i,j) == 0.0);
        REQUIRE(b(i,j) == 0.0);
    }
  }

  multi_array<double,2> c({3,3});
  set_identity(c);

  multi_array<float,2> d({3,3});
  set_identity(d);

  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      if(i==j){
        REQUIRE(c(i,i) == 1.0);
        REQUIRE(d(i,i) == 1.0);
      }else{
        REQUIRE(c(i,j) == 0.0);
        REQUIRE(d(i,j) == 0.0);
      }
      }
    }

  }

  SECTION("A*B multiplication"){
    multi_array<double,2> A({2,3});
    multi_array<double,2> B({3,4});
    multi_array<double,2> AB({2,4});
    multi_array<double,2> RAB({2,4});

    for(int i = 0; i<2; i++){
      for(int j = 0; j<3; j++){
        A(i,j) = i + 2*j;
      }
    }

    for(int i = 0; i<3; i++){
      for(int j = 0; j<4; j++){
        B(i,j) = i + 3*j;
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

    matmul(A,B,AB);
    REQUIRE((bool)(RAB == AB));

  }

  SECTION("At*B multiplication"){
    multi_array<double,2> A({3,2});
    multi_array<double,2> B({3,4});
    multi_array<double,2> AtB({2,4});
    multi_array<double,2> RAtB({2,4});

    for(int i = 0; i<3; i++){
      for(int j = 0; j<2; j++){
        A(i,j) = j + 2*i;
      }
    }

    for(int i = 0; i<3; i++){
      for(int j = 0; j<4; j++){
        B(i,j) = i + 3*j;
      }
    }

    set_zero(RAtB);
    for(int i = 0; i<2; i++){
      for(int j = 0; j<4; j++){
        for(int k = 0; k< 3; k++){
            RAtB(i,j) += A(k,i)*B(k,j);
        }
      }
    }

    matmul_transa(A,B,AtB);
    REQUIRE((bool)(RAtB == AtB));

  }

  SECTION("A*Bt multiplication"){
    multi_array<double,2> A({2,3});
    multi_array<double,2> B({4,3});
    multi_array<double,2> ABt({2,4});
    multi_array<double,2> RABt({2,4});

    for(int i = 0; i<2; i++){
      for(int j = 0; j<3; j++){
        A(i,j) = i + 2*j;
      }
    }
    for(int i = 0; i<4; i++){
      for(int j = 0; j<3; j++){
        B(i,j) = j + 3*i;
      }
    }

    set_zero(RABt);
    for(int i = 0; i<2; i++){
      for(int j = 0; j<4; j++){
        for(int k = 0; k< 3; k++){
            RABt(i,j) += A(i,k)*B(j,k);
        }
      }
    }

    matmul_transb(A,B,ABt);
    REQUIRE((bool)(RABt == ABt));
  }

  SECTION("At*Bt multiplication"){
    multi_array<double,2> A({3,2});
    multi_array<double,2> B({4,3});
    multi_array<double,2> AtBt({2,4});
    multi_array<double,2> RAtBt({2,4});

    for(int i = 0; i<3; i++){
      for(int j = 0; j<2; j++){
        A(i,j) = j + 2*i;
      }
    }
    for(int i = 0; i<4; i++){
      for(int j = 0; j<3; j++){
        B(i,j) = j + 3*i;
      }
    }

    set_zero(RAtBt);
    for(int i = 0; i<2; i++){
      for(int j = 0; j<4; j++){
        for(int k = 0; k< 3; k++){
            RAtBt(i,j) += A(k,i)*B(j,k);
        }
      }
    }

    matmul_transab(A,B,AtBt);
    REQUIRE((bool)(RAtBt == AtBt));

  }


}
