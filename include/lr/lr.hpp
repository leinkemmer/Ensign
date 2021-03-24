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
        // TODO
    }

    lr2(Index r, array<Index,2> N, stloc sl=stloc::host) {
        // TODO
    }

    Index problem_size() {
        return X.shape()[0];
    }
};


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

template<class T>
void init(lr2& lr, vector<T*> X, vector<T*> V, std::function<T(T*,T*)> inner_product) {
    Index rank = X.size();
    
    //lr.X.setIdentity();
    //lr.V.setIdentity();
    //lr.S.setZero();
    // TODO
    fill(lr.S.begin(), lr.S.end(), T(0.0));

    for(Index k=0;k<rank;k++) {
        double l2_X = sqrt(inner_product(X[k], X[k]));
        double l2_V = sqrt(inner_product(V[k], V[k]));

        for(Index i=0;i<lr.problem_size();i++) {
            lr.X(i, k) = X[k][i]/l2_X;
            lr.V(i, k) = V[k][i]/l2_V;
            lr.S(k, k) = l2_X*l2_V;
        }
    }

    
    multi_array<T, 2> X_R(lr.S.shape()), V_R(lr.S.shape());
    gram_schmidt(lr.X, X_R, inner_product);
    gram_schmidt(lr.V, V_R, inner_product);

    //lr.S = X_R*lr.S*V_R.transpose()
    // TODO
}

