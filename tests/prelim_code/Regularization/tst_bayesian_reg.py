import numpy as np
from numpy.linalg import cholesky
import scipy.optimize as optimize
from functools import partial


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


# Tikhonov regularization
def tikhonov_simple(X, u_glob, lambda_2):
    aa = X.shape[1]
    c = np.ones(aa)
    C = np.diag(c)

    f_glob = np.linalg.inv(X.T @ X + lambda_2 * C) @ (X.T @ u_glob)
    return f_glob
