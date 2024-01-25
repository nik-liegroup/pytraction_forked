import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd, norm, cholesky
from tests.prelim_code.prelim_utilis import *
from tests.prelim_code.tst_example_fields import *
from scipy.linalg import norm


def bayesian_regularization(X, u_glob, ux, uy):
    # Concatenate displacement vector components in noise window
    noise_window = 10
    ux, uy = ux[noise_window:noise_window], uy[noise_window:noise_window]
    noise_vec = np.array([ux.flatten(), uy.flatten()])

    # Calculate inverse noise variance
    beta = 1 / np.var(noise_vec)

    # Standardize the input data along columns to their spread
    sd_X = np.std(X, axis=0)
    X = (X - np.mean(X, axis=0)) / sd_X
    u_glob = u_glob - np.mean(u_glob)

    # Singular Value Decomposition of X where U and V are orthogonal matrices and S is diagonal with singular values
    U, S, V = svd(X, full_matrices=False, compute_uv=True)

    # Matrix multiplication X.T @ X = V @ (S.T @ S) @ V.T
    XX = X.T @ X

    # Prepare identity matrix
    aa = X.shape[1]
    c = np.ones(aa)
    C = np.diag(c)

    # Parameters for Golden Section Search
    alpha1 = 200  # initial left alpha
    alpha2 = 40000  # initial right alpha
    step_size = 30
    max_n = 200

    # Golden Section search to find maximum of log evidence
    n = 0
    nn = []
    lambdas = []

    while True:
        # Find middle value of alpha1 and alpha2
        middle = 0.5 * (alpha1 + alpha2)

        # Calculate evidence for alpha value closer to alpha1 and for closer to alpha2
        middle_up = middle + 0.5 * step_size
        evidence_up = log_evidence(middle_up, beta, U, S, V, u_glob, C, X, aa, XX)
        middle_down = middle - 0.5 * step_size
        evidence_down = log_evidence(middle_down, beta, U, S, V, u_glob, C, X, aa, XX)

        # Choose value yielding smaller evidence to reduce the possible interval for alpha
        if evidence_up >= evidence_down:
            alpha1 = middle_down
        else:
            alpha2 = middle_up

        n += 1
        nn.append(n)
        lambdas.append(alpha1 / beta)

        if norm(alpha1 - alpha2) / norm(alpha1) < 1e-5 or n == max_n:
            break

    # Calculation of final solution with optimal parameters
    f_glob = tikhonov_simple(X, u_glob, lambdas[-1])

    # Undo standardization
    f_glob /= sd_X[:, np.newaxis]

    return f_glob


def log_evidence(alpha, beta, U, S, V, u_glob, C, X, aa, XX):
    lambd = alpha / beta
    f_glob = tikhonov_simple(X, u_glob, lambd)

    # Calculate log(det(A))
    A = alpha * C + beta * XX
    L = cholesky(A)
    log_det_A = 2 * np.sum(np.log(np.diag(L)))

    # Formula for log evidence
    evidence_value = -0.5 * alpha * f_glob.T @ f_glob - 0.5 * beta * (X @ f_glob - u_glob).T @ (X @ f_glob - u_glob) \
                     - 0.5 * log_det_A + 0.5 * aa * np.log(beta) + 0.5 * aa * np.log(alpha) \
                     - 0.5 * aa * np.log(2 * np.pi)

    return evidence_value


# Tikhonov regularization
def tikhonov(X, u_glob, lambda_2):
    aa = X.shape[1]
    c = np.ones(aa)
    C = np.diag(c)

    f_glob = np.linalg.inv(X.T @ X + lambda_2 * C) @ (X.T @ u_glob)
    return f_glob


def tikhonov2(U, s, V, b, lambd, x_0=None):
    """
    Computes the Tikhonov regularized solution x_lambda, given the singular value decomposition SVD or the general
    singular value decomposition GSVD.  If the SVD is used, i.e. if U, s, and V are specified, then standard-form
    regularization is applied:
        min { || A x - b ||^2 + lambda^2 || x - x_0 ||^2 } .
    If, on the other hand, the GSVD is used, i.e. if U, sm = [sigma,mu], and X are specified, then general-form
    regularization is applied:
        min { || A x - b ||^2 + lambda^2 || L (x - x_0) ||^2 } .
    Note that x_0 cannot be used if A is underdetermined and L ~= I.
    If lambda is a vector, then x_lambda is a matrix such that x_lambda = [ x_lambda(1), x_lambda(2), ... ] .

    Returns:
    The solution norm (standard-form case) or seminorm (general-form case) and the residual norm are returned in eta
    and rho.

    Per Christian Hansen, DTU Compute, April 14, 2003.
    Reference: A. N. Tikhonov & V. Y. Arsenin, "Solutions of Ill-Posed Problems", Wiley, 1977.
    """

    # Initialization
    if np.min(lambd) < 0:
        raise ValueError('Illegal regularization parameter lambda')

    m, n = U.shape[0], V.shape[0]
    p, ps = s.shape
    beta = U[:, :p].T @ b
    zeta = s * beta
    ll = len(lambd)
    x_lambda = np.zeros((n, ll))
    rho = np.zeros((ll, 1))
    eta = np.zeros((ll, 1))

    if ps == 1:
        # The standard-form case
        for i in range(ll):
            if x_0 is None:
                x_lambda[:, i] = np.squeeze(V[:, :p] @ (zeta / (s ** 2 + lambd[i] ** 2)))
                rho[i] = lambd[i] ** 2 * np.linalg.norm(beta / (s ** 2 + lambd[i] ** 2))
            else:
                omega = V.T @ x_0
                x_lambda[:, i] = np.squeeze(V[:, :p] @ ((zeta + lambd[i] ** 2 * omega) / (s ** 2 + lambd[i] ** 2)))
                rho[i] = lambd[i] ** 2 * np.linalg.norm((beta - s * omega) / (s ** 2 + lambd[i] ** 2))
            eta[i] = np.linalg.norm(x_lambda[:, i])

        if U.shape[0] > p:
            rho = np.sqrt(rho ** 2 + np.linalg.norm(b - U[:, :n] @ np.concatenate([beta, U[:, p:n].T @ b])) ** 2)
    else:
        raise ValueError('Please provide a 1-dim array containing the singular values in descending order.')

    return x_lambda, rho, eta


# Example usage of tikhonov with SVD for a non-singular matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U, S, V = np.linalg.svd(A)
b = np.array([1, 2, 3]).reshape(3, 1)
lamba = np.array([0.1, 0.01])
x_0 = np.zeros((3, 1))

[x_lambda, rho, eta] = tikhonov2(U, S.reshape(3, 1), V, b, lamba, x_0)


# Define parameters
elastic_modulus = 1000
s = 0.48
pix_per_mu = 1
beta = 0.1
lambda_2 = 0.0001

sigma = 5
x0, y0 = 5, 5
width_x, width_y = 4, 4

point_dens = 50
x_min, y_min = -10, -10
x_max, y_max = 10, 10

# Create grid of points in spatial coordinates
x_val, y_val = np.linspace(x_min, x_max, point_dens), np.linspace(y_min, y_max, point_dens)
xx, yy = np.meshgrid(x_val, y_val, indexing='xy')

# Calculate BEM matrix
gamma_glob = traction_bem(xx, yy, 'conv', point_dens, s, elastic_modulus)

# Define forward Fredholm term f(x, y) under integral
forward_fx, forward_fy, forward_glob_norm = tri_pole(xx, yy, x0, y0, sigma)
forward_glob_f = np.concatenate([forward_fx.reshape(point_dens ** 2, 1),
                                 forward_fy.reshape(point_dens ** 2, 1)])

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_ux, inverse_uy, inverse_norm = vortex(xx, yy, x0, y0)
inverse_glob_u = np.concatenate([inverse_ux.reshape(point_dens ** 2, 1),
                                 inverse_uy.reshape(point_dens ** 2, 1)])

# Calculate forward solution
bem_forward_u = (gamma_glob @ forward_glob_f)
bem_forward_ux = bem_forward_u[:point_dens ** 2].reshape(point_dens, point_dens).T
bem_forward_uy = bem_forward_u[point_dens ** 2:].reshape(point_dens, point_dens).T

# Calculate inverse solution
bem_inverse_f = bayesian_regularization(gamma_glob, inverse_glob_u, inverse_ux, inverse_uy)
bem_inverse_fx = bem_inverse_f[:point_dens ** 2].reshape(point_dens, point_dens).T
bem_inverse_fy = bem_inverse_f[point_dens ** 2:].reshape(point_dens, point_dens).T

# Plots
# Create subplot for forward solution
fig_forward, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Boundary element method: Forward solution')

# Quiver plot for the first vector field
im = axs[0].imshow(np.rot90(np.sqrt(forward_fx ** 2 + forward_fy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[0].quiver(xx, yy, forward_fx, forward_fy, color='black')
cbar = fig_forward.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

# Quiver plot for the second vector field
im = axs[1].imshow(np.rot90(np.sqrt(bem_forward_ux ** 2 + bem_forward_uy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(xx, yy, bem_forward_ux, bem_forward_uy, color='black')
axs[0].set_axis_off()
axs[1].set_axis_off()
cbar = fig_forward.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Displacement field [\u03bcm]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

fig_forward.savefig('bem_forward.png', dpi=300, bbox_inches="tight")
axs[0].set_title('Input: 2D traction field f(x,y)')
axs[1].set_title('BEM forward solution u(x,y) to Fredholm integral')

# Create subplot for inverse solution
fig_inverse, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Boundary element method: Inverse solution')

# Quiver plot for the third vector field
im = axs[0].imshow(np.rot90(np.sqrt(inverse_ux ** 2 + inverse_uy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[0].quiver(xx, yy, inverse_ux, inverse_uy, color='black')
cbar = fig_inverse.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Displacement field [\u03bcm]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

# Quiver plot for the fourth vector field
im = axs[1].imshow(np.rot90(np.sqrt(bem_inverse_fx ** 2 + bem_inverse_fy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(xx, yy, bem_inverse_fx, bem_inverse_fy, color='black')
axs[0].set_axis_off()
axs[1].set_axis_off()
cbar = fig_inverse.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

fig_inverse.savefig('bem_inverse.png', dpi=300, bbox_inches="tight")
axs[0].set_title('Input: 2D displacement field u(x,y)')
axs[1].set_title('BEM inverse solution f(x,y) to Fredholm integral')

plt.suptitle('Boundary element method')
plt.tight_layout()
plt.show()
