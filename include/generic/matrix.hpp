#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>


template<class T>
void set_zero(multi_array<T,2>& a);

template<class T>
void set_identity(multi_array<T,2>& a);

template<class T>
void matmul(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

template<class T>
void matmul_transa(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

template<class T>
void matmul_transb(const multi_array<T,2>& a, const multi_array<T,2>& b, multi_array<T,2>& c);

//template<class T>
//void transpose(const multi_array<T,2>& a, multi_array<T,2>& b);

