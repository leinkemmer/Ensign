#pragma once

#include <iostream>
#include <vector>

#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <generic/timer.hpp>
#include <generic/netcdf.hpp>
#include <lr/coefficients.hpp>
#include <lr/lr.hpp>

#include <netcdf.h>

namespace Ensign{

#ifdef __OPENMP__
#pragma omp declare reduction(+ : Ensign::multi_array<double, 2> : omp_out += omp_in)  \
    initializer(omp_priv = decltype(omp_orig)(omp_orig))
#endif

#ifdef __OPENMP__
#pragma omp declare reduction(+ : Ensign::multi_array<double, 4> : omp_out += omp_in)  \
    initializer(omp_priv = decltype(omp_orig)(omp_orig))
#endif

// General classes for the hierarchical DLR approximation
// TODO: introduce a template parameter `N` for arbitrary many outgoing legs
template <class T> struct node {
    const std::string id;

    node* const parent;
    std::array<node*, 2> child;
    const Index n_basisfunctions;

    Ensign::multi_array<T, 2> S;

    node() = default;

    node(const std::string _id, node* const _parent, std::array<node*, 2> _child,
         const Index _r_in, const Index _n_basisfunctions)
        : id(_id), parent(_parent), child(_child), n_basisfunctions(_n_basisfunctions),
          S({_r_in, _r_in})
    {
        assert(n_basisfunctions <= _r_in);
    }

    virtual ~node() = default;

    virtual bool IsInternal() const = 0;
    virtual bool IsExternal() const = 0;
    virtual void Initialize(int ncid) = 0;

    Index RankIn() const { return S.shape()[0]; }
};

template <class T> struct internal_node : virtual node<T> {
    Ensign::multi_array<T, 3> Q;
    Ensign::multi_array<T, 3> G;

    internal_node(const std::string _id, internal_node* const _parent,
                  const Index _r_in, const std::array<Index, 2> _r_out,
                  const Index _n_basisfunctions)
        : node<T>(_id, _parent, {nullptr, nullptr}, _r_in, _n_basisfunctions),
          Q({_r_out[0], _r_out[1], _r_in}), G({_r_out[0], _r_out[1], _r_in})
    {
    }

    bool IsInternal() const override { return true; }

    bool IsExternal() const override { return false; }

    std::array<Index, 2> RankOut() const
    {
        return array<Index, 2>({Q.shape()[0], Q.shape()[1]});
    }

    void Initialize(int ncid) override;

    void Write(int ncid, int id_r_in, std::array<int, 2> id_r_out) const;

    Ensign::multi_array<T, 2> ortho(const T weight,
                                            const Ensign::blas_ops& blas);
};

template <class T> struct external_node : virtual node<T> {
    Ensign::multi_array<T, 2> X;

    external_node(const std::string _id, internal_node<T>* const _parent,
                  const Index _dx, const Index _r_in, const Index _n_basisfunctions)
        : node<T>(_id, _parent, {nullptr, nullptr}, _r_in, _n_basisfunctions),
          X({_dx, _r_in})
    {
    }

    bool IsInternal() const override { return false; }

    bool IsExternal() const override { return true; }

    Index ProblemSize() const { return X.shape()[0]; }

    void Initialize(int ncid) override;

    void Write(int ncid, int id_r_in, int id_dx) const;

    Ensign::multi_array<T, 2> ortho(const T weight,
                                            const Ensign::blas_ops& blas);
};

template <class T>
Ensign::multi_array<T, 2> internal_node<T>::ortho(const T weight,
                                                          const Ensign::blas_ops& blas)
{
    Ensign::multi_array<T, 2> Qmat({Ensign::prod(RankOut()), node<T>::RankIn()});
    Ensign::multi_array<T, 2> Q_R({node<T>::RankIn(), node<T>::RankIn()});
    Ensign::Tensor::matricize<2>(Q, Qmat);
    Q_R = Tensor::ortho(Qmat, node<T>::n_basisfunctions, weight, blas);
    Ensign::Tensor::tensorize<2>(Qmat, Q);

    return Q_R;
};

template <class T>
Ensign::multi_array<T, 2> external_node<T>::ortho(const T weight,
                                                          const Ensign::blas_ops& blas)
{
    Ensign::multi_array<T, 2> X_R({node<T>::RankIn(), node<T>::RankIn()});
    X_R = Tensor::ortho(X, node<T>::n_basisfunctions, weight, blas);

    return X_R;
};

} // namespace Ensign