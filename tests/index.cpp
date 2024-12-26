#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <generic/index.hpp>


TEST_CASE("incr_vec_index", "[incr_vec_index]")
{
    std::array<Index, 5> input = {0, 1, 1, 1, 3};
    std::array<Index, 5> interval = {1, 2, 3, 4, 5};

    std::array<Index, 5> output1_ref = {0, 0, 2, 1, 3};
    std::array<Index, 5> output2_ref = {0, 0, 0, 2, 3};
    std::array<Index, 5> output3_ref = {0, 0, 0, 0, 4};

    Ensign::IndexFunction::incr_vec_index(std::begin(interval), std::begin(input),
                                          std::end(input));
    REQUIRE(bool(input == output1_ref));

    Ensign::IndexFunction::incr_vec_index(std::begin(interval), std::begin(input),
                                          std::end(input));
    Ensign::IndexFunction::incr_vec_index(std::begin(interval), std::begin(input),
                                          std::end(input));
    REQUIRE(bool(input == output2_ref));

    input = {0, 1, 2, 3, 3};
    Ensign::IndexFunction::incr_vec_index(std::begin(interval), std::begin(input),
                                          std::end(input));
    REQUIRE(bool(input == output3_ref));
}

TEST_CASE("vec_index_to_comb_index", "[vec_index_to_comb_index]")
{
    std::vector<Index> vec_index(10);
    std::vector<Index> interval(10);
    Index comb_index;
    Index comparison_index = 592088944020;
    for (Index i = 0; i < 10; ++i) {
        vec_index[i] = i;
        interval[i] = 20 - i;
    }
    comb_index = Ensign::IndexFunction::vec_index_to_comb_index(
        std::begin(vec_index), std::end(vec_index), std::begin(interval));
    REQUIRE(bool(comb_index == comparison_index));
}

TEST_CASE("comb_index_to_vec_index", "[comb_index_to_vec_index]")
{
    Index comb_index = 23084307895;
    std::vector<Index> interval(10);
    std::vector<Index> vec_index(10);
    std::vector<Index> comparison_vec(10);
    for (Index i = 0; i < 10; ++i) {
        interval[i] = 11;
        comparison_vec[i] = i;
    }
    Ensign::IndexFunction::comb_index_to_vec_index(
        comb_index, std::begin(interval), std::begin(vec_index), std::end(vec_index));
    REQUIRE(bool(vec_index == comparison_vec));

    comb_index = 79;
    vec_index.resize(4);
    comparison_vec.resize(4);
    interval = {4, 2, 3, 5};
    comparison_vec = {3, 1, 0, 3};
    Ensign::IndexFunction::comb_index_to_vec_index(
        comb_index, std::begin(interval), std::begin(vec_index), std::end(vec_index));
    REQUIRE(bool(vec_index == comparison_vec));
}