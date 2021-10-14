#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <lr/lr.hpp>
#include <generic/matrix.hpp>

TEST_CASE( "Low rank structure 2D", "[low_rank]" ) {

  blas_ops blas;

  SECTION("Gram-Schmidt"){
    multi_array<double,2> A({4,3});
    multi_array<double,2> R({3,3});

    A(0,0) = 1.0; A(0,1) = 3.0; A(0,2) = 5.0;
    A(1,0) = 2.0; A(1,1) = 0.0; A(1,2) = 1.0;
    A(2,0) = 4.0; A(2,1) = 4.0; A(2,2) = 8.0;
    A(3,0) = 2.0; A(3,1) = 5.0; A(3,2) = 7.0;

    multi_array<double,2> Ac;
    Ac = A;

    std::function<double(double*,double*)> ip = inner_product_from_const_weight(1.0, 4);

    gram_schmidt gs(&blas);
    gs(A, R, ip);

    multi_array<double,2> id({3,3});
    set_identity(id);

    multi_array<double,2> R1({3,3});
    multi_array<double,2> R2({4,3});

    blas_ops blas;
    blas.matmul_transa(A,A,R1);
    REQUIRE(bool(R1==id));

    blas.matmul(A,R,R2);
    REQUIRE(bool(R2 == Ac));

  }

  SECTION("Initialization low rank structure"){

    array<double,4> x1 = {1.0,2.0,4.0,2.0};
    array<double,4> x2 = {3.0,0.0,4.0,5.0};
    array<double,4> x3 = {5.0,1.0,8.0,7.0};

    array<double,4> v1 = {1.0,5.0,2.0,3.0};
    array<double,4> v2 = {2.0,6.0,5.0,4.0};
    array<double,4> v3 = {1.0,2.0,3.0,15.0};


    vector<const double*> X;
    X.push_back(x1.begin());
    X.push_back(x2.begin());
    X.push_back(x3.begin());

    vector<const double*> V;
    V.push_back(v1.begin());
    V.push_back(v2.begin());
    V.push_back(v3.begin());


    std::function<double(double*,double*)> ip = inner_product_from_const_weight(1.0, 4);

    lr2<double> lr0(3,{4,4});

    blas_ops blas;
    initialize(lr0, X, V, ip, ip, blas);

    multi_array<double,2> id({3,3});
    set_identity(id);

    multi_array<double,2> R1({3,3});

    blas.matmul_transa(lr0.X,lr0.X,R1);
    REQUIRE(bool(R1 == id));

    blas.matmul_transa(lr0.V,lr0.V,R1);
    REQUIRE(bool(R1 == id));

    multi_array<double,2> Rtmp({4,3});
    multi_array<double,2> RRR({4,4});

    multi_array<double,2> RR({4,4});

    blas.matmul(lr0.X,lr0.S,Rtmp);
    blas.matmul_transb(Rtmp,lr0.V,RRR);

    multi_array<double,2> XX({4,3});
    multi_array<double,2> VV({4,3});

      for(int i=0; i <4;i++){
        XX(i,0) = x1[i];
        VV(i,0) = v1[i];
      }
      for(int i=0; i <4;i++){
        XX(i,1) = x2[i];
        VV(i,1) = v2[i];
      }
      for(int i=0; i <4;i++){
        XX(i,2) = x3[i];
        VV(i,2) = v3[i];
      }

    blas.matmul_transb(XX,VV,RR);

    REQUIRE(bool(RR == RRR));


  }
}
