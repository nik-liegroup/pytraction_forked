import numpy as np
from functools import partial
import scipy.optimize as optimize
from scipy.sparse import csr_matrix, spdiags

from pytraction.utils import sparse_cholesky, ft_2Dvector_field
from pytraction.inversion import traction_fourier


def minus_logevidence(
    alpha: float,
    pos: np.ndarray,
    vec_u: np.ndarray,
    gamma_glob,
    beta: float,
    E,
    s,
    scaling_z
):
    """
    Calculate minus logevidence.
    """
    kxx, kyy, ft_ux, ft_uy, _, _ = ft_2Dvector_field(pos, vec_u)

    # Combine x- and y-Fourier components into a new row array and transpose
    u_glob = np.array([ft_ux.flatten(), ft_uy.flatten()]).flatten()
    u_glob = np.expand_dims(u_glob, axis=1)

    # Create identity matrix
    gamma_dim = gamma_glob.shape[0]
    id = spdiags(data=np.ones((gamma_dim)), diags=(0), m=gamma_dim, n=gamma_dim)

    # Collaps matrix products
    XX = csr_matrix(gamma_glob).T * csr_matrix(gamma_glob)
    BX = (2 * beta / gamma_dim) * csr_matrix(XX)
    C = (2 / gamma_dim) * id
    pre_factor = gamma_dim * np.log(beta) - gamma_dim * np.log(2 * np.pi)

    # Calculate lambda value
    lambd = alpha / beta

    # Get fourier transforms of traction field
    _, _, ft_fx, ft_fy, _, _, _, _, _ = traction_fourier(pos=pos,
                                                       vec=vec_u,
                                                       s=s,
                                                       elastic_modulus=E,
                                                       lambd=lambd,
                                                       scaling_z=scaling_z,
                                                       zdepth=0,
                                                       slim=True)

    f_glob = np.array([ft_fx.flatten(), ft_fy.flatten()]).flatten()
    f_glob = np.expand_dims(f_glob, axis=1)

    A = alpha * csr_matrix(C) + BX

    L = sparse_cholesky(csr_matrix(A)).toarray()
    logdetA = 2 * np.sum(np.log(np.diag(L)))

    gamma_dim = gamma_glob.shape
    Xf_u = gamma_glob * f_glob - u_glob
    idx = Xf_u.shape[0] // 2
    Ftux1 = Xf_u[:idx]
    Ftuy1 = Xf_u[idx:]

    ff = np.sum(np.sum(ft_fx * np.conj(ft_fx) + ft_fy * np.conj(ft_fy))) / (0.5 * gamma_dim[1])
    uu = np.sum(np.sum(Ftux1 * np.conj(Ftux1) + Ftuy1 * np.conj(Ftuy1))) / (0.5 * gamma_dim[0])

    evidence_value = np.real(
        -0.5 * (-alpha * ff - beta * uu - logdetA + gamma_dim[1] * np.log(alpha) + pre_factor)
    )
    return evidence_value


def optimal_lambda(
    pos: np.ndarray, vec_u: np.ndarray, beta: float, E: float, s:float, scaling_z:float, gamma_glob
):
    """
    Calculate optimal lambda value.
    """
    # Define interval boundaries for alpha value
    alpha1 = 1e-6
    alpha2 = 1e6

    # Define target function to be minimized
    target = partial(
        minus_logevidence,
        pos=pos,
        vec_u=vec_u,
        beta=beta,
        E=E,
        s=s,
        scaling_z=scaling_z,
        gamma_glob = gamma_glob,
    )

    # Golden section search to find alpha at minimum of -log(Evidence)
    alpha_opt = optimize.fminbound(target, alpha1, alpha2, disp=3)

    if (alpha_opt > alpha1 * 0.9) and (alpha_opt < 0.9 * alpha2): # Optimal alpha value is not close to boundary
        evidence_one = -target(alpha_opt)
        lambd = alpha_opt / beta

    else:
        # Center interval around optimal alpha
        alpha1 = 1e-3 * alpha_opt
        alpha2 = 1e3 * alpha_opt

        # Create alpha function on interval
        alpha = np.linspace(alpha1, alpha2, 100)
        logevidence = alpha.copy()
        for index, value in np.ndenumerate(alpha):
            logevidence[index] = minus_logevidence(
                alpha=value,
                pos=pos,
                vec_u=vec_u,
                beta=beta,
                E=E,
                s=s,
                scaling_z=scaling_z,
                gamma_glob=gamma_glob,
            )

        # Calculate angle of logevidence(alpha) derivatives to alpha-axis from gradient
        grad_logevidence = np.gradient(logevidence, alpha)
        angle_logevidence = -np.rad2deg(np.arctan(grad_logevidence))

        # Find alpha value where local slope of logevidence is 45deg to alpha-axis
        angle_diff = np.abs(angle_logevidence - 45)
        idx = angle_diff.argmin()
        alpha_opt = alpha[idx]

        # Calculate evidence and lambda
        evidence_one = -target(alpha_opt)
        lambd = alpha_opt / beta

        msg = f'Alpha value too close to boundary, found new evidence value: {evidence_one} with new alpha value: \
        {alpha_opt} using gradient method.'
        raise RuntimeWarning(msg)

    return lambd, evidence_one


def tikhonov_simple(gamma_glob, vec_u, lambd):
    """
    Tikhonov regularization.
    """
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
