import time
from functools import partial

import numpy as np
import scipy.optimize as optimize
from scipy.sparse import csr_matrix, spdiags

from pytraction.fourier import reg_fourier_tfm
from pytraction.utils import sparse_cholesky


def minus_logevidence(
    alpha,
    beta,
    C_a,
    BX_a,
    X,
    fuu,
    constant,
    Ftux,
    Ftuy,
    E,
    s,
    cluster_size,
    kx,
    ky
):
    i_max = Ftux.shape[0]
    j_max = Ftuy.shape[1]

    aa = X.shape
    LL = alpha / beta
    _, _, _, _, Ftfx, Ftfy = reg_fourier_tfm(
        Ftux, Ftuy, kx, ky, LL, E, s, cluster_size, i_max, j_max, slim=True
    )
    fxx = Ftfx.reshape(i_max * j_max, 1)
    fyy = Ftfy.reshape(i_max * j_max, 1)

    f = np.array([fxx, fyy]).T.flatten()
    f = np.expand_dims(f, axis=1)

    A = alpha * csr_matrix(C_a) + BX_a

    L = sparse_cholesky(csr_matrix(A)).toarray()
    logdetA = 2 * np.sum(np.log(np.diag(L)))

    Xf_u = X * f - fuu
    idx = Xf_u.shape[0] // 2
    Ftux1 = Xf_u[:idx]
    Ftuy1 = Xf_u[idx:]

    ff = np.sum(np.sum(Ftfx * np.conj(Ftfx) + Ftfy * np.conj(Ftfy))) / (0.5 * aa[1])
    # ff = np.real(ff)
    uu = np.sum(np.sum(Ftux1 * np.conj(Ftux1) + Ftuy1 * np.conj(Ftuy1))) / (0.5 * aa[0])
    # uu = np.real(uu)

    evidence_value = np.real(
        -0.5 * (-alpha * ff - beta * uu - logdetA + aa[1] * np.log(alpha) + constant)
    )
    return evidence_value


def optimal_lambda(
    beta, ftux, ftuy, kx, ky, E, s, cluster_size, X
):
    i_max = ftux.shape[0]
    j_max = ftuy.shape[1]

    # Reshaped into column vectors
    fux1 = ftux.reshape(i_max * j_max, 1)
    fux2 = ftuy.reshape(i_max * j_max, 1)

    # Combine x- and y-Fourier components into a row with resulting dim(2 * i_max * j_max)
    fuu = np.array([fux1, fux2]).T.flatten()

    # Add additional dimension to array for further processing
    fuu = np.expand_dims(fuu, axis=1)

    aa = X.shape
    c = np.ones((aa[1]))
    C = spdiags(c, (0), aa[1], aa[1])
    XX = csr_matrix(X).T * csr_matrix(X)
    BX_a = beta * csr_matrix(XX) / aa[1] * 2
    C_a = C / aa[1] * 2
    constant = aa[0] * np.log(beta) - aa[0] * np.log(2 * np.pi)

    # Golden section search method to find alpha at minimum of -log(Evidence)
    # setting the range of parameter search. Change if minimum can not be found in your data
    alpha1 = 1e-6
    alpha2 = 1e6

    target = partial(
        minus_logevidence,
        beta=beta,
        C_a=C_a,
        BX_a=BX_a,
        X=X,
        fuu=fuu,
        constant=constant,
        Ftux=ftux,
        Ftuy=ftuy,
        E=E,
        s=s,
        cluster_size=cluster_size,
        kx=kx,
        ky=ky
    )

    alpha_opt = optimize.fminbound(target, alpha1, alpha2, disp=3)

    if (alpha_opt > alpha1 * 0.9) and (alpha_opt < 0.9 * alpha2):
        evidence_one = -target(alpha_opt)
        lambda_2 = alpha_opt / beta
    else:
        alpha1 = 1e-6
        alpha2 = 0.05
        alpha = np.linspace(alpha1, alpha2, 100)
        logevidence = alpha.copy()

        for index, value in np.ndenumerate(alpha):
            logevidence[index] = minus_logevidence(
                alpha=value,
                beta=beta,
                C_a=C_a,
                BX_a=BX_a,
                X=X,
                fuu=fuu,
                constant=constant,
                Ftux=ftux,
                Ftuy=ftuy,
                E=E,
                s=s,
                cluster_size=cluster_size,
                kx=kx,
                ky=ky
            )

        grad_logevidence = np.gradient(logevidence, alpha)
        angle_logevidence = -np.rad2deg(np.arctan(grad_logevidence))

        angle_diff = np.abs(angle_logevidence - 45)
        idx = angle_diff.argmin()
        alpha_opt = alpha[idx]

        #Default value
        alpha_opt = 0.05

        ###
        import matplotlib.pyplot as plt
        plt.plot(alpha, logevidence, 'g')
        plt.axvline(x=alpha_opt, color='b')
        #plt.show()
        plt.close()
        ###

        evidence_one = -target(alpha_opt)
        lambda_2 = alpha_opt / beta

        print(f'Alpha value too close to boundary, found new evidence value: {evidence_one} with new alpha value: {alpha_opt} using gradient method')

    return lambda_2, None, evidence_one
