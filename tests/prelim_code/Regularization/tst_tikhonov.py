import numpy as np
import time


def tikhonov_hansen(A, b, lambd, x_0=None):
    """
    Computes the Tikhonov regularized solution x_lambda (standard form), given a singular value decomposition SVD
    (U, s, and V) of matrix A:
        min { || A x - b ||^2 + lambda^2 || x - x_0 ||^2 } .
    If lambda is a vector, then x_lambda is a matrix such that x_lambda = [ x_lambda(1), x_lambda(2), ... ] .

    Returns:
    The solution norm (standard-form) and the residual norm are returned in eta and rho.

    Per Christian Hansen, DTU Compute, April 14, 2003.
    Reference: A. N. Tikhonov & V. Y. Arsenin, "Solutions of Ill-Posed Problems", Wiley, 1977.
    """

    start = time.time()

    U, s, V = np.linalg.svd(A)
    s = s.reshape(-1, 1)

    if np.min(lambd) < 0:
        raise ValueError('Illegal regularization parameter lambda')

    m, n = U.shape[0], V.shape[0]
    p, ps = s.shape
    beta = U[:, :p].T @ b
    zeta = s * beta
    ll = len(lambd) if isinstance(lambd, np.ndarray) else 1
    x_lambda = np.zeros((n, ll))
    rho = np.zeros((ll, 1))
    eta = np.zeros((ll, 1))

    if ps == 1:
        # The standard-form case
        for i in range(ll):
            lambd_i = lambd[i] if isinstance(lambd, np.ndarray) else lambd
            if x_0 is None:
                x_lambda[:, i] = np.squeeze(V[:, :p] @ (zeta / (s ** 2 + lambd_i ** 2)))
                rho[i] = lambd_i ** 2 * np.linalg.norm(beta / (s ** 2 + lambd_i ** 2))
            else:
                omega = V.T @ x_0
                x_lambda[:, i] = np.squeeze(V[:, :p] @ ((zeta + lambd_i ** 2 * omega) / (s ** 2 + lambd_i ** 2)))
                rho[i] = lambd_i ** 2 * np.linalg.norm((beta - s * omega) / (s ** 2 + lambd_i ** 2))
            eta[i] = np.linalg.norm(x_lambda[:, i])

        if U.shape[0] > p:
            rho = np.sqrt(rho ** 2 + np.linalg.norm(b - U[:, :n] @ np.concatenate([beta, U[:, p:n].T @ b])) ** 2)
    else:
        raise ValueError('Please provide a 1-dim array containing the singular values in descending order.')

    end = time.time()
    print(f"tikhonov_hansen: {end - start}")

    return x_lambda, rho, eta


# Tikhonov regularization
def tikhonov_simple(X, u_glob, lambda_2):
    start = time.time()
    aa = X.shape[1]
    c = np.ones(aa)
    C = np.diag(c)

    f_glob = np.linalg.inv(X.T @ X + lambda_2 * C) @ (X.T @ u_glob)

    end = time.time()
    print(f"tikhonov_simple: {end - start}")
    return f_glob


# Example usage of tikhonov with SVD for a non-singular matrix
A = np.random.rand(500, 500)


b = np.random.rand(500, 1)
lamba = 0.1
x_0 = None

x_lambda1, rho1, eta1 = tikhonov_hansen(A, b, lamba, x_0)
x_lambda2 = tikhonov_simple(A, b, lamba)

print(f"Methods equality: {np.allclose(x_lambda1, x_lambda2, rtol=1e-05)}")
