import numpy as np
from numpy.linalg import cholesky
import scipy.optimize as optimize
from functools import partial
from pytraction.inversion import traction_fourier
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse


def log_evidence(alpha, beta, X, u_glob):
    # Calculate traction field given current lambda
    lambd = alpha / beta
    f_glob = tikhonov_simple(X, u_glob, lambd)

    # Create identity matrix from X
    m, n = X.shape
    Id = np.eye(m, n)

    # Calculate cholesky decomposition for hessian matrix A
    A = alpha * Id + beta * X.T @ X
    L = cholesky(A)
    log_detA = 2 * np.sum(np.log(np.diag(L)))

    # Formula for log evidence
    evidence_value = - 0.5 * alpha * f_glob.T @ f_glob \
                     - 0.5 * beta * (X @ f_glob - u_glob).T @ (X @ f_glob - u_glob) \
                     - 0.5 * log_detA \
                     - m * np.log(2 * np.pi) \
                     + m * np.log(beta) \
                     + n * np.log(alpha)

    minus_evidence = - evidence_value.item()

    return minus_evidence


def log_evidence_ft(alpha, beta, C_a, BX_a, X, fuu, constant, elastic_modulus, s, pos, vec):
    aa = X.shape
    lambd = alpha / beta
    fx, fy, ft_fx, ft_fy, ft_ux, ft_uy, kxx, kyy, X = traction_fourier(
        pos, vec, s, elastic_modulus, lambd=lambd, scaling_factor=None, zdepth=0, slim=True
    )

    i_max = (np.shape(kxx)[0])
    j_max = (np.shape(kyy)[1])

    fxx = ft_fx.reshape(i_max * j_max, 1)
    fyy = ft_fy.reshape(i_max * j_max, 1)

    f = np.array([fxx, fyy]).T.flatten()
    f = np.expand_dims(f, axis=1)

    A = alpha * csr_matrix(C_a) + BX_a
    A = csr_matrix(A)

    LU = splinalg.splu(A.tocsc(), diag_pivot_thresh=0)  # sparse LU decomposition

    L = LU.L.dot(sparse.diags(LU.U.diagonal() ** 0.5)).tocsr()
    L = L.toarray()

    logdetA = 2 * np.sum(np.log(np.diag(L)))

    Xf_u = X * f - fuu
    idx = Xf_u.shape[0] // 2
    Ftux1 = Xf_u[:idx]
    Ftuy1 = Xf_u[idx:]

    ff = np.sum(np.sum(ft_fx * np.conj(ft_fx) + ft_fy * np.conj(ft_fy))) / (0.5 * aa[1])
    # ff = np.real(ff)
    uu = np.sum(np.sum(Ftux1 * np.conj(Ftux1) + Ftuy1 * np.conj(Ftuy1))) / (0.5 * aa[0])
    # uu = np.real(uu)

    evidence_value = np.real(
        -0.5 * (-alpha * ff - beta * uu - logdetA + aa[1] * np.log(alpha) + constant)
    )
    return evidence_value


def bayesian_regularization(X, u_glob, beta):
    # Standardize the differential operator X along its columns to their spread
    X_sd = np.std(X, axis=0)
    X_mean = np.mean(X, axis=0)
    X = (X - X_mean) / X_sd

    # Standardize the deformation field u to its mean
    u_glob = (u_glob - np.mean(u_glob))

    # Parameters for Golden Section Search
    alpha_left = 1e-6  # Initial left alpha
    alpha_right = 1e6  # Initial right alpha

    # Set target function for optimization
    target_func = partial(
        log_evidence,
        beta=beta,
        X=X,
        u_glob=u_glob
    )
    alpha_opt, fval, ierr, numfunc = optimize.fminbound(target_func, alpha_left, alpha_right, disp=3, full_output=True)

    lambd = alpha_opt / beta
    print(f"Final regularization lambda: {lambd}")
    return lambd


def bayesian_regularization_ft(
    beta, ftux, ftuy, elastic_modulus, s, i_max, j_max, X, pos, vec
):
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
    # setting the range of parameter search. Change if maximum can not be found in your data
    alpha1 = 1e-6
    alpha2 = 1e6

    target = partial(
        log_evidence_ft,
        beta=beta,
        C_a=C_a,
        BX_a=BX_a,
        X=X,
        fuu=fuu,
        constant=constant,
        elastic_modulus=elastic_modulus,
        s=s,
        pos=pos,
        vec=vec
    )
    alpha_opt = optimize.fminbound(target, alpha1, alpha2, disp=3)

    evidence_one = -target(alpha_opt)
    lambda_2 = alpha_opt / beta

    return lambda_2, None, evidence_one


# Tikhonov regularization
def tikhonov_simple(gamma_glob, vec_u, lambd):
    ux = vec_u[:, :, 0]
    uy = vec_u[:, :, 1]
    i_max = ux.shape[0]
    j_max = uy.shape[1]
    u = np.array([ux.flatten(), uy.flatten()]).flatten()

    aa = gamma_glob.shape[1]
    c = np.ones(aa)
    C = np.diag(c)

    f = np.linalg.inv(gamma_glob.T @ gamma_glob + lambd * C) @ (gamma_glob.T @ u)
    fx = f[:i_max*j_max].reshape(i_max, j_max).T
    fy = f[i_max * j_max:].reshape(i_max, j_max).T
    return fx, fy
