#pragma once

#include <iterator>
#include <vector>

#include <generic/common.hpp>
#include <generic/storage.hpp>

namespace Ensign {

namespace IndexFunction {
template <class InputIt, class InputItInt>
Index vec_index_to_comb_index(InputIt first, InputIt last, InputItInt first_int)
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
void comb_index_to_vec_index(Index comb_index, InputItInt first, OutputIt d_first,
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
void incr_vec_index(InputIt first, OutputIt d_first, OutputIt d_last)
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

} // namespace Ensign