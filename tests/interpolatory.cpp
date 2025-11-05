#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <lr/interpolatory.hpp>

using namespace Ensign;
using namespace Ensign::Matrix;

blas_ops blas;


TEST_CASE( "DEIM oversampling", "[deim_overs]" ) {

  SECTION("deim_overs") {

    multi_array<double,2> U({7,3});
    // From the python script deim_ext_experiments.py
    U(0,0)=-0.46456425951148805; U(1,0)=-0.11802692122703781; U(2,0)=-0.7152630424064148; U(3,0)=-0.29330991515549404; U(4,0)=-0.3737151696481189; U(5,0)=-0.06469518556994265; U(6,0)=-0.16961507429516318;
    U(0,1)=0.008225016101339879; U(1,1)=-0.8138882892617407; U(2,1)=0.3435864136099434; U(3,1)=0.10334799784755648; U(4,1)=-0.4274077071926592; U(5,1)=-0.133376785929551; U(6,1)=-0.09120884959095332;
    U(0,2)=-0.004422178016274861; U(1,2)=-0.32092235191221874; U(2,2)=-0.24184937349678717; U(3,2)=0.18337230728520634; U(4,2)=0.13357107641681468; U(5,2)=0.8235802964675081; U(6,2)=0.3297681437922386;


    multi_array<Index,1> I({5});
    deim_ext(U, 5, I);

    REQUIRE(I(0) == 1);
    REQUIRE(I(1) == 2);
    REQUIRE(I(2) == 5);
    REQUIRE(I(3) == 4);
    REQUIRE(I(4) == 0);

  }
}


TEST_CASE( "Interpolatory LR schemes", "[interp]" ) {

  array<double,5> x1 = {1.0,2.0,4.0,2.0,-2.0};
  array<double,5> x2 = {1e-2*3.0,0.0,1e-2*4.0,1e-2*5.0,1e-2*3.0};

  array<double,4> v1 = {1.0,5.0,2.0,3.0};
  array<double,4> v2 = {2.0,6.0,5.0,4.0};

  vector<const double*> X;
  X.push_back(x1.begin());
  X.push_back(x2.begin());

  vector<const double*> V;
  V.push_back(v1.begin());
  V.push_back(v2.begin());


  SECTION("colloquation_to_lr"){
    Index nx=5, ny=4;
    Index r=2, r_over=3;
    
    lr2<double> lr0(r,{nx,ny});
    initialize(lr0, X, V, 1.0, 1.0, blas);

    multi_array<Index, 1> I({r_over}), J({r_over});
    deim_ext(lr0.X, r_over, I);
    deim_ext(lr0.V, r_over, J);
    
    // colloquate the square of the matrix 
    multi_array<double,2> f_full = lr0.full(blas);
    multi_array<double,2> f_I({r_over,ny});
    for(Index m=0;m<3;m++) {
      for(Index j=0;j<ny;j++) {
        f_I(m, j) = pow(f_full(I(m), j), 2);
      }
    }

    multi_array<double,2> f_J({nx,r_over});
    for(Index m=0;m<r_over;m++) {
      for(Index i=0;i<nx;i++) {
        f_J(i, m) = pow(f_full(i, J(m)), 2);
      }
    }


    // to LR without orthogonalization
    lr2<double> f1(r, {nx,ny});
    colloquation_to_lr(f_I, f_J, I, f1.X, f1.V, blas);
    f1.S.set_zero();
    f1.S(0,0)=1.0; f1.S(1,1)=1.0;
    multi_array<double,2> f1_full = f1.full(blas);

    // to LR with orthogonalization
    lr2<double> f2(r, {nx,ny});
    colloquation_to_lr(f_I, f_J, I, f2, blas);
    multi_array<double,2> f2_full = f2.full(blas);

    double err1=0.0, err2=0.0;
    for(Index j=0;j<ny;j++) {
      for(Index i=0;i<nx;i++) {
        double exact = pow(f_full(i,j), 2);
        err1 = max_err(err1, abs(exact-f1_full(i,j)));
        err2 = max_err(err2, abs(exact-f2_full(i,j)));
      }
    }

    cout << "err1: " << err1 << " " << "err2: " << err2 << endl;
    REQUIRE(err1 < 2e-3);
    REQUIRE(err2 < 2e-3);
  }

}
