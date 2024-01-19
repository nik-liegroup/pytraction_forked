from typing import Tuple

import numpy as np
from openpiv import widim

from pytraction.fourier import fourier_xu, reg_fourier_tfm
from pytraction.optimal_lambda import optimal_lambda
from pytraction.utils import align_slice, remove_boarder_from_aligned, interp_vec2grid
from pytraction.postprocess import strain_energy, contraction_moments, contraction_moments_ft


def iterative_piv(img: np.ndarray,
                  ref: np.ndarray,
                  config
                  ):  # ToDo: Problems with defining output types as Tuple[...]
    """
    Perform iterative PIV on drift corrected images and returns drift corrections (dx,dy) and displacement vectors (u,v)
    for positions (x,y).
    """
    dx, dy, img = align_slice(img, ref)  # Get drift in x,y and drift corrected img
    if config.config["settings"]["crop_aligned_slice"]:
        img, ref = remove_boarder_from_aligned(img, ref)  # Crop img and ref to remove black borders
    stack = np.stack([img, ref])  # Creates stack from img and ref
    x, y, u, v, mask = compute_piv(img, ref, config)

    return x, y, u, v, (stack, dx, dy)


def compute_piv(img: np.ndarray,
                ref: np.ndarray,
                config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PIV using a window displacement iterative method with implementation of window size coarsening.
    """
    try:
        # Compute PIV using the window displacement iterative method implemented by openpiv
        x, y, u, v, mask = widim.WiDIM(  # Returns displacement vectors (u,v) for positions (x,y)
            ref.astype(np.int32),
            img.astype(np.int32),
            np.ones_like(ref).astype(np.int32),  # Mark whole image to be used for computation
            **config.config["piv"],  # Unpack and pass the contents of a dictionary as keyword arguments
        )
        return x, y, u, v, mask  # Mask from widim.WiDIM function call
    except Exception as e:
        if isinstance(e, ZeroDivisionError):
            config.config["piv"]["min_window_size"] = (
                config.config["piv"]["min_window_size"] // 2
            )
            print(
                f"Reduced min window size to {config.config['piv']['min_window_size']} in recursive call"
            )
            return compute_piv(img, ref, config)
        else:
            raise e


def calculate_traction_map(pos: np.array,
                           vec: np.array,
                           beta: float,
                           meshsize: float,
                           s: float,
                           pix_per_mu: float,
                           E: float):
    """
    Calculates the traction map given the displacement vector field and the noise value beta.
    """
    # Interpolate displacement field onto rectangular grid using meshsize
    grid_mat, u, i_max, j_max = interp_vec2grid(pos, vec, meshsize, [])

    # Transform displacement field to fourier space
    ftux, ftuy, kxx, kyy, i_max, j_max, X = fourier_xu(u, i_max, j_max, E, s, meshsize)

    # Calculate lambda from bayesian model
    L, evidencep, evidence_one = optimal_lambda(
        beta, ftux, ftuy, kxx, kyy, E, s, meshsize, i_max, j_max, X, 1
    )

    # Calculate traction field in fourier space and transform back to spatial domain
    f_pos, f_nm_2, f_magnitude, f_n_m, ftfx, ftfy = reg_fourier_tfm(
        ftux, ftuy, kxx, kyy, L, E, s, meshsize, i_max, j_max, pix_per_mu, 0, grid_mat
    )

    # off with the shapes flip back into position
    traction_magnitude = f_magnitude.reshape(i_max, j_max).T
    traction_magnitude = np.flip(traction_magnitude, axis=0)

    # Extract positional and vector coordinates
    xx = grid_mat[:, :, 0]
    yy = grid_mat[:, :, 1]
    uxx = u[:, :, 0]
    uyy = u[:, :, 1]
    txx = f_n_m[:, :, 0]
    tyy = f_n_m[:, :, 1]

    # Calculate strain energy
    energy = strain_energy(xx, yy, txx, tyy, uxx, uyy, pix_per_mu)

    return traction_magnitude, f_n_m, energy, L, txx, tyy
