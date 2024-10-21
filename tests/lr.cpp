#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <lr/lr.hpp>
#include <generic/matrix.hpp>

using namespace Ensign;
using namespace Ensign::Matrix;

blas_ops blas;

template<class IP>
void orthogonalize_unit_test(IP inner_product){
  multi_array<double,2> A({4,3});
  multi_array<double,2> R({3,3});

  A(0,0) = 1.0; A(0,1) = 3.0; A(0,2) = 5.0;
  A(1,0) = 2.0; A(1,1) = 0.0; A(1,2) = 1.0;
  A(2,0) = 4.0; A(2,1) = 4.0; A(2,2) = 8.0;
  A(3,0) = 2.0; A(3,1) = 5.0; A(3,2) = 7.0;

  multi_array<double,2> Ac;
  Ac = A;

  orthogonalize gs(&blas);
  gs(A, R, inner_product);

  multi_array<double,2> id({3,3});
  set_identity(id);

  multi_array<double,2> R1({3,3});
  multi_array<double,2> R2({4,3});

  blas_ops blas;
  blas.matmul_transa(A,A,R1);
  REQUIRE(bool(R1==id));

  blas.matmul(A,R,R2);
  REQUIRE(bool(R2 == Ac));
};

template void orthogonalize_unit_test(std::function<double(double*,double*)> inner_product);
template void orthogonalize_unit_test(double inner_product);
template void orthogonalize_unit_test(double* inner_product);



void orthogonalize_unit_test_gpu(double inner_product){
  multi_array<double,2> _A({4,3});
  multi_array<double,2> _R({3,3});

  _A(0,0) = 1.0; _A(0,1) = 3.0; _A(0,2) = 5.0;
  _A(1,0) = 2.0; _A(1,1) = 0.0; _A(1,2) = 1.0;
  _A(2,0) = 4.0; _A(2,1) = 4.0; _A(2,2) = 8.0;
  _A(3,0) = 2.0; _A(3,1) = 5.0; _A(3,2) = 7.0;

  multi_array<double,2> _Ac;
  _Ac = _A;

  multi_array<double,2> A({4,3},stloc::device);
  multi_array<double,2> R({3,3},stloc::device);

  A = _A;

  orthogonalize gs(&blas);
  gs(A, R, inner_product);

  multi_array<double,2> id({3,3});
  set_identity(id);

  multi_array<double,2> R1({3,3});
  multi_array<double,2> R2({4,3});

  _A = A;
  _R = R;

  blas_ops blas;
  blas.matmul_transa(_A,_A,R1);
  REQUIRE(bool(R1==id));

  blas.matmul(_A,_R,R2);
  REQUIRE(bool(R2 == _Ac));
};

TEST_CASE( "Low rank structure 2D", "[low_rank]" ) {

  SECTION("Gram-Schmidt"){

    std::function<double(double*,double*)> ip = inner_product_from_const_weight(1.0, 4);

    orthogonalize_unit_test(ip);

  }

  SECTION("Householder constant weight, GPU"){

  double ip = 1.0;

  #ifdef __CUDACC__

  cout << "Test for GPU"<< endl;
  orthogonalize_unit_test_gpu(ip);
  cout << "Test for GPU over"<< endl;
  
  #else

  orthogonalize_unit_test(ip);

  #endif

  }

  SECTION("Householder constant vector weight"){

    multi_array<double,1> ip({4});
    ip(0) = 1.0; ip(1) = 1.0; ip(2) = 1.0; ip(3) = 1.0;

    orthogonalize_unit_test(ip.data());

  }
    
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
  


  SECTION("Initialization low rank structure"){
    
    lr2<double> lr0(3,{4,4});
    std::function<double(double*,double*)> ip = inner_product_from_const_weight(1.0, 4);
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

  array<double,4> x21 = {0.5,0.4,1.0,2.0};
  array<double,4> x22 = {0.7,0.3,0.4,7.0};

  array<double,4> v21 = {2.0,3.0,1.0,3.0};
  array<double,4> v22 = {2.0,1.0,2.0,2.0};

  vector<const double*> X2;
  X2.push_back(x21.begin());
  X2.push_back(x22.begin());

  vector<const double*> V2;
  V2.push_back(v21.begin());
  V2.push_back(v22.begin());

  SECTION("Add two low-rank representations") {
    
    std::function<double(double*,double*)> ip = inner_product_from_const_weight(0.25, 4);

    lr2<double> lr0(3,{4,4});
    initialize(lr0, X, V, ip, ip, blas);

    lr2<double> lr1(2,{4,4});
    initialize(lr1, X2, V2, ip, ip, blas);

    lr2<double> out(5,{4,4});
    lr_add(1.5, lr0, -0.5, lr1, out, ip, ip, blas);

    multi_array<double,2> lr0_full = lr0.full(blas);
    multi_array<double,2> lr1_full = lr1.full(blas);
    multi_array<double,2> exact({4,4});
    for(Index j=0;j<exact.shape()[1];j++)
      for(Index i=0;i<exact.shape()[0];i++)
        exact(i,j) = 1.5*lr0_full(i,j) - 0.5*lr1_full(i,j);

    REQUIRE(bool(out.full(blas)==exact)); // check without truncation


    lr2<double> out_truncated(4,{4,4});
    lr_truncate(out, out_truncated, blas);
    multi_array<double,2> out_truncated_full = out_truncated.full(blas);

    double err = 0.0;
    for(Index j=0;j<4;j++)
      for(Index i=0;i<4;i++)
        if(std::isnan(out_truncated_full(i,j))) 
          err = std::numeric_limits<double>::infinity();
        else
          err = max(err, std::abs(out_truncated_full(i,j)-exact(i,j))); 

    cout << "add truncation error: " << err << endl;
    REQUIRE( err < 1e-10 );

  }

  SECTION("Multiply two low-rank representations") {

    std::function<double(double*,double*)> ip = inner_product_from_const_weight(0.25, 4);

    lr2<double> lr0(3,{4,4});
    initialize(lr0, X, V, ip, ip, blas);

    lr2<double> lr1(2,{4,4});
    initialize(lr1, X2, V2, ip, ip, blas);

    lr2<double> out(6,{4,4});
    lr_mul(lr0, lr1, out, ip, ip, blas);

    multi_array<double,2> lr0_full = lr0.full(blas);
    multi_array<double,2> lr1_full = lr1.full(blas);
    multi_array<double,2> out_full = out.full(blas);
    multi_array<double,2> exact({4,4});
    for(Index j=0;j<exact.shape()[1];j++)
      for(Index i=0;i<exact.shape()[0];i++) {
        exact(i,j) = lr0_full(i,j)*lr1_full(i,j);
        cout << exact(i,j) - out_full(i,j) << endl;
      }
    
    REQUIRE(bool(out.full(blas)==exact)); // check without truncation


    lr2<double> out_truncated(4,{4,4});
    lr_truncate(out, out_truncated, blas);
    multi_array<double,2> out_truncated_full = out_truncated.full(blas);

    double err = 0.0;
    for(Index j=0;j<4;j++)
      for(Index i=0;i<4;i++)
        if(std::isnan(out_truncated_full(i,j))) 
          err = std::numeric_limits<double>::infinity();
        else
          err = max(err, std::abs(out_truncated_full(i,j)-exact(i,j))); 

    cout << "mul truncation error: " << err << endl;
    REQUIRE( err < 1e-10 );
  }

  SECTION("Compute inner products of low-rank representations") {
    double w = 0.25;
    std::function<double(double*,double*)> ip = inner_product_from_const_weight(w, 4);

    lr2<double> lr0(3,{4,4});
    initialize(lr0, X, V, ip, ip, blas);

    lr2<double> lr1(2,{4,4});
    initialize(lr1, X2, V2, ip, ip, blas);
    
    multi_array<double,2> lr0_full = lr0.full(blas);
    multi_array<double,2> lr1_full = lr1.full(blas);
    double expected_ip = 0.0, expected_norm_sq = 0.0;
    for(Index j=0;j<4;j++) {
      for(Index i=0;i<4;i++) {
        expected_ip += w*w*lr0_full(i,j)*lr1_full(i,j);
        expected_norm_sq += w*w*pow(lr0_full(i,j),2);
      }
    }

    double lr_ip = lr_inner_product(lr0, lr1, w*w, blas);
    double norm_sq = lr_norm_sq(lr0, blas);

    cout << "norm err: " << abs(expected_norm_sq-norm_sq) << endl;
    cout << "inner product error: " << abs(expected_ip-lr_ip) << endl;
    REQUIRE( abs(expected_norm_sq - norm_sq) < 1e-10 );
    REQUIRE( abs(expected_ip - lr_ip) < 1e-10 );
  }

}
