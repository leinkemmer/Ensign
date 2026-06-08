#include <lr/interpolatory.hpp>
#include <Eigen/Dense>

namespace Ensign {

void deim_ext(const multi_array<double,2>& _U, Index r_over, multi_array<Index,1>& p) {
    using namespace Eigen;

    Index np = _U.shape()[1]; // number of columns in U
    MatrixXd U(_U.shape()[0], np);
    for(Index m=0;m<np;m++) {
        for(Index i=0;i<U.rows();i++) {
            U(i, m) = _U(i, m);
        }
    }

    // First pivot
    Index maxIdx;
    U.col(0).cwiseAbs().maxCoeff(&maxIdx);
    p(0) = maxIdx;

    // Select the first np indices by DEIM
    for(Index n = 1; n < np; n++) {
        // Build submatrix U[p[:n], :n]
        MatrixXd Up(n, n);
        for (Index i = 0; i < n; ++i) {
            for (Index j = 0; j < n; ++j) {
                Up(i, j) = U(p(i), j);
            }
        }

        VectorXd b(n);
        for(Index i = 0; i < n; ++i)
            b(i) = U(p(i), n);


        VectorXd c = Up.lu().solve(b);

        // Compute residual
        VectorXd r = (U.col(n) - U.leftCols(n)*c).cwiseAbs();

        // Find new pivot
        r.maxCoeff(&maxIdx);
        p(n) = maxIdx;
    }

    std::sort(p.begin(), p.begin()+np);

    // add indices for oversampling (if r>np)
    for(Index j=np;j<r_over;j++) {
        // U[p, :]
        MatrixXd Usub(j, U.cols());
        for (Index i = 0; i < (int)j; ++i)
            Usub.row(i) = U.row(p(i));

        JacobiSVD<MatrixXd> svd(Usub, ComputeThinU | ComputeThinV);
        VectorXd S = svd.singularValues();
        MatrixXd W = svd.matrixV();

        double g = pow(S(S.size() - 2), 2) - pow(S(S.size() - 1), 2);
        MatrixXd Ub = W.transpose() * U.transpose();

        ArrayXd r = ArrayXd::Constant(Ub.cols(), g) + Ub.array().square().colwise().sum().transpose();
        r = r - (r.square() - 4 * g * Ub.row(Ub.rows() - 1).array().square().transpose()).sqrt();

        // Sort r descending
        vector<std::pair<double, Index>> vals;
        for (Index i = 0; i < r.size(); i++) {
            vals.push_back({r(i), i});
        }
        std::sort(vals.begin(), vals.end(), [](std::pair<double, Index>& a, std::pair<double, Index>& b){ return a.first > b.first; });

        // Find first index not in p
        Index newIdx = -1;
        for (auto& v : vals) {
            if (std::find(p.begin(), p.begin()+j, v.second) == p.begin()+j) {
                newIdx = v.second;
                break;
            }
        }
        if (newIdx != -1) {
            p(j) = newIdx;
        } else {
            cout << "ERROR: could not determine an oversampling index" << endl;
            exit(1);
        }
    }
}


double colloquation_to_lr(const multi_array<double,2>& f_I, multi_array<double,2>& f_J, const multi_array<Index,1>& I, multi_array<double,2>& X, multi_array<double,2>& L, Matrix::blas_ops& blas) {
    Index nx = X.shape()[0];
    Index nv = L.shape()[0];
    Index r = X.shape()[1];
    Index r_over = f_I.shape()[0];

    multi_array<double,2> U_hat({nx,r_over}), V_hat({r_over,r_over});
    multi_array<double,1> sigma_hat({r_over});
    svd(f_J, U_hat, V_hat, sigma_hat, blas);
    //cout << "sigma_hat: " << sigma_hat << endl;

    for(Index m=0;m<r;m++) {
        for(Index i=0;i<nx;i++) {
            X(i, m) = U_hat(i, m);
        }
    }
    multi_array<double,2> U_trunc({r_over,r});
    for(Index m=0;m<r;m++) {
        for(Index M=0;M<r_over;M++) {
            U_trunc(M,m) = U_hat(I(M),m);
        }
    }

    // compute the pseudo-inverse by SVD
    multi_array<double,2> U1({r_over, r});
    multi_array<double,2> V1({r, r});
    multi_array<double,1> sigma1({r});
    svd(U_trunc, U1, V1, sigma1, blas);

    double cond = sigma1(0)/(sigma1(r-1)+1e-15);
    if(cond > 1e7) {
        cout << "WARNING: condition number of U in colloquation_to_lr is " << cond << endl;
    }

   // cout << "V1: " << V1 << endl;

    multi_array<double,2> tmp({r, nv});
    blas.matmul_transa(U1, f_I, tmp);
    double tol = r_over*1e-15*sigma1(0);
    //cout << tol << " " << "sigma1: " << sigma1 <<  endl;
    for(Index m=0;m<r;m++) {
        for(Index i=0;i<nv;i++) {
            if(sigma1(m) > tol) {
                tmp(m, i) /= sigma1(m);
            } else {
                tmp(m, i) = 0.0;
            }
        }
    }
    //cout << "tmp: " << tmp << endl;

    multi_array<double,2> L_transpose({r,nv});
    blas.matmul(V1, tmp, L_transpose);
    blas.transpose(L_transpose, L);

    return cond;
}

double  colloquation_to_lr(const multi_array<double,2>& f_I, multi_array<double,2>& f_J, const multi_array<Index,1>& I, lr2<double>& f, Matrix::blas_ops& blas) {
    double cond = colloquation_to_lr(f_I, f_J, I, f.X, f.V, blas);

    orthogonalize ortho(&blas);
    multi_array<double,2> R({f.rank(),f.rank()});
    ortho(f.V, f.S, 1.0);
    Matrix::transpose_inplace(f.S);
    return cond;
}

}