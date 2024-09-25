#pragma once

#include <iterator>
#include <vector>

#include <generic/storage.hpp>

#ifdef __OPENMP__
#include <omp.h>
#endif

namespace IndexFunction {
template <class InputIt, class InputItInt>
Index VecIndexToCombIndex(InputIt first, InputIt last, InputItInt first_int)
{
    Index comb_index = 0;
    Index stride = 1;
    for (; first != last; ++first, ++first_int) {
        comb_index += *first * stride;
        stride *= *first_int;
    }
    return comb_index;
}

template <class InputItInt, class OutputIt>
void CombIndexToVecIndex(Index comb_index, InputItInt first, OutputIt d_first,
                         OutputIt d_last)
{
    assert(d_first != d_last);
    for (; d_first != std::next(d_last, -1); ++first, ++d_first) {
        *d_first = comb_index % *first;
        comb_index = Index(comb_index / *first);
    }
    *(std::next(d_last, -1)) = comb_index;
}

template <class InputIt, class OutputIt>
void IncrVecIndex(InputIt first, OutputIt d_first, OutputIt d_last)
{
    assert(d_first != d_last);
    for (; d_first != std::next(d_last, -1); ++first, ++d_first) {
        ++(*d_first);
        if (*d_first < *first)
            return;
        *d_first = typename std::iterator_traits<OutputIt>::value_type(0);
    }
    ++(*(std::next(d_last, -1)));
}

} // namespace IndexFunction