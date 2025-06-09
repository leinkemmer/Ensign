#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cmath>

#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <generic/tensor.hpp>
#include <lr/coefficients.hpp>
#include <lr/lr.hpp>

class generator {
  private:
    Index i = 0;

  public:
    inline Index operator()() { return ++i; };
};

TEST_CASE("Matricize_Tensorize", "[Matricize_Tensorize]")
{
    Index d0 = 5, d1 = 7, d2 = 9;
    Ensign::multi_array<Index, 3> ten({d0, d1, d2}), ten2({d0, d1, d2}),
        ten1({d0, d1, d2}), ten0({d0, d1, d2});
    Ensign::multi_array<Index, 2> mat2({d0 * d1, d2}), mat2_ref({d0 * d1, d2});
    Ensign::multi_array<Index, 2> mat1({d2 * d0, d1}), mat1_ref({d2 * d0, d1});
    Ensign::multi_array<Index, 2> mat0({d1 * d2, d0}), mat0_ref({d1 * d2, d0});

    std::generate(std::begin(ten), std::end(ten), generator{});

    Ensign::Tensor::matricize<2>(ten, mat2);
    Ensign::Tensor::matricize<1>(ten, mat1);
    Ensign::Tensor::matricize<0>(ten, mat0);

    generator generator2{};
    for (Index k = 0; k < d2; ++k) {
        for (Index j = 0; j < d1; ++j) {
            for (Index i = 0; i < d0; ++i) {
                mat2_ref(i + j * d0, k) = generator2();
            }
        }
    }

    generator generator1{};
    for (Index k = 0; k < d2; ++k) {
        for (Index j = 0; j < d1; ++j) {
            for (Index i = 0; i < d0; ++i) {
                mat1_ref(i + k * d0, j) = generator1();
            }
        }
    }

    generator generator0{};
    for (Index k = 0; k < d2; ++k) {
        for (Index j = 0; j < d1; ++j) {
            for (Index i = 0; i < d0; ++i) {
                mat0_ref(j + k * d1, i) = generator0();
            }
        }
    }

    REQUIRE(bool(mat2 == mat2_ref));
    REQUIRE(bool(mat1 == mat1_ref));
    REQUIRE(bool(mat0 == mat0_ref));

    Ensign::Tensor::tensorize<2>(mat2, ten2);
    Ensign::Tensor::tensorize<1>(mat1, ten1);
    Ensign::Tensor::tensorize<0>(mat0, ten0);

    REQUIRE(bool(ten2 == ten));
    REQUIRE(bool(ten1 == ten));
    REQUIRE(bool(ten0 == ten));
}

TEST_CASE("RemoveElement", "[RemoveElement]")
{
    std::array<Index, 10> start_vec;
    std::array<Index, 9> vec1, vec2, vec3, vec1_ref, vec2_ref, vec3_ref;

    std::generate(std::begin(start_vec), std::end(start_vec), generator{});

    Ensign::remove_element(std::begin(start_vec), std::end(start_vec), std::begin(vec1),
                           0);
    Ensign::remove_element(std::begin(start_vec), std::end(start_vec), std::begin(vec2),
                           5);
    Ensign::remove_element(std::begin(start_vec), std::end(start_vec), std::begin(vec3),
                           9);

    vec1_ref = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    vec2_ref = {1, 2, 3, 4, 5, 7, 8, 9, 10};
    vec3_ref = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    REQUIRE(bool(vec1 == vec1_ref));
    REQUIRE(bool(vec2 == vec2_ref));
    REQUIRE(bool(vec3 == vec3_ref));
}

TEST_CASE("Orthogonalize", "[Orthogonalize]")
{
    Index r = 5, dx = 6;
    Index n_basisfunctions = 1;
    Ensign::multi_array<double, 2> mat({dx, r}), mat_ref({dx, r});
    Ensign::Matrix::set_zero(mat);
    Ensign::Matrix::set_zero(mat_ref);
    mat(0, 0) = 1.0;
    std::function<double(double*, double*)> ip;
    Ensign::Matrix::blas_ops blas;

    Ensign::multi_array<double, 2> Q(mat), R({r, r});
    R = Ensign::Tensor::orthogonalize(Q, n_basisfunctions, 1.0, blas);

    Ensign::multi_array<double, 2> Q2({r, r}), id_r({r, r});
    Ensign::Matrix::set_identity(id_r);
    blas.matmul_transa(Q, Q, Q2);
    blas.matmul(Q, R, mat_ref);

    REQUIRE(bool(Q2 == id_r));
    REQUIRE(bool(mat == mat_ref));
}