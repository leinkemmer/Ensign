#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>
#include <generic/matrix.hpp>

template<class T, size_t d1, size_t d2>
void coeff(multi_array<T,d1>& a, multi_array<T,d2>& b, T w, multi_array<T,d1+d2-2>& out);

template<class T, size_t d1, size_t d2>
void coeff(multi_array<T,d1>& a, multi_array<T,d2>& b, T* w, multi_array<T,d1+d2-2>& out);

template<class T, size_t d1>
void coeff_one(multi_array<T,d1>& a, T* w, multi_array<T,1>& out);

template<class T, size_t d1>
void coeff_one(multi_array<T,d1>& a, T w, multi_array<T,1>& out);

/*
template<class T, size_t d1, size_t d2>
void coeff_rho(multi_array<T,d1>& a, T* w, multi_array<T,1>& out);

template<class T, size_t d1, size_t d2>
void coeff_rho(multi_array<T,d1>& a, T w, multi_array<T,1>& out);
*/

/*
template<class T, size_t d1>
void coeff3(multi_array<T,2>& a, multi_array<T,2>& b, multi_array<T,d1>& c, T w, multi_array<T,2>& out);

template<class T>
void coeff3(multi_array<T,2>& a, multi_array<T,2>& b,  multi_array<T,1>& c, T* w, multi_array<T,2>& out);
*/
