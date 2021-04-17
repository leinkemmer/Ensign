#pragma once

#include <generic/common.hpp>
#include <generic/storage.hpp>

#include <functional>

template<class T>
struct lr2 {
    multi_array<T, 2> S;
    multi_array<T, 2> X;
    multi_array<T, 2> V;

    lr2(stloc sl=stloc::host) {
      multi_array<T,2> _S(sl);
      multi_array<T,2> _X(sl);
      multi_array<T,2> _V(sl);
      S = _S;
      X = _X;
      V = _V;
    }

    lr2(Index r, array<Index,2> N, stloc sl=stloc::host) {
      multi_array<T,2> _S({r,r},sl);
      multi_array<T,2> _X({N[0],r},sl);
      multi_array<T,2> _V({N[1],r},sl);
      S = _S;
      X = _X;
      V = _V;
    }

    Index problem_size_X() {
        return X.shape()[0];
    }
    Index problem_size_V() {
        return V.shape()[0];
    }
};

template<class T>
void initialize(lr2<T>& lr, vector<T*> X, vector<T*> V, int n_b, std::function<T(T*,T*)> inner_product_X, std::function<T(T*,T*)> inner_product_V);

template<class T>
std::function<T(T*,T*)> inner_product_from_weight(T* w, Index N);

template<class T>
std::function<T(T*,T*)> inner_product_from_const_weight(T w, Index N);

//template<class T>
//void ptw_prod(T* v, Index n, T scal);

//template<class T>
//void ptw_diff(T* v1, Index n, T* v2);

template<class T>
void ptw_div(T* v, Index n, T scal);

template<class T>
void gram_schmidt(multi_array<T,2>& Q, multi_array<T,2>& R,
        std::function<T(T*,T*)> inner_product);

/*
template<class T>
struct lr3 {
    multi_array<T, 3> S;
    multi_array<T, 2> X;
    multi_array<T, 2> Y;
    multi_array<T, 2> Z;

};

template<class T>
struct lrh23 {
    multi_array<T, 2> S;
    lr3 X;
    lr3 V;
};
*/
/*
template<class T>
std::function<T(T*,T*)> inner_product_from_weight(T* w, Index N) {
    return [N](T* a, T*b) {
        T result=T(0.0);
        for(Index i=0;i<N;i++)
            result += w[i]*a[i]*b[i];
        return result;
    }
}

template<class T>
std::function<T(T*,T*)> l2_inner_product(T* w, T w) {
    return [w](T* a, T*b) {
        T result=T(0.0);
        for(Index i=0;i<N;i++)
            result += w*a[i]*b[i];
        return result;
    }
}

*/
/*
MatrixXd gram_schmidt(const MatrixXd& _V, MatrixXd& R,
        function<double(const VectorXd&,const VectorXd&)> inner_product) {


    // do modified Gram-Schmidt (necessary for f0v!=1)
    int r = _V.cols();
    R.resize(r,r);

    MatrixXd V = _V;
    MatrixXd v = V;
    R.setZero();
    for(int i=0;i<r;i++) {
        V.col(i) = v.col(i);
        for(int j=0;j<i;j++) {
            R(j,i) = inner_product(V.col(j), v.col(i));
            V.col(i) -= R(j,i)*V.col(j);
        }
        double nrm = sqrt(inner_product(V.col(i), V.col(i)));

        if(nrm < 1e-13) {
            VectorXd nv(V.rows());
            for(Index k=0;k<Index(nv.size());k++)
                nv[i] = cos(2.0*M_PI*i*k/double(nv.size()));
            v.col(i) = nv;
            i--; // redo the current step with artifical data
        } else {
            V.col(i) = V.col(i)/nrm;
            R(i,i) = inner_product(v.col(i),V.col(i));
        }
    }

    return V;
}*/
