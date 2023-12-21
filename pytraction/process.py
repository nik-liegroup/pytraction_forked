from typing import Tuple

import numpy as np
from openpiv import widim

from pytraction.fourier import fourier_xu, reg_fourier_tfm
from pytraction.optimal_lambda import optimal_lambda
from pytraction.utils import align_slice, remove_boarder_from_aligned


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
    # Transform displacement field to Fourier space
    grid_mat, i_max, j_max, X, fuu, Ftux, Ftuy, u = fourier_xu(
        pos, vec, meshsize, 1, s, []
    )

    # get lambda from baysian bad boi
    L, evidencep, evidence_one = optimal_lambda(
        beta, fuu, Ftux, Ftuy, 1, s, meshsize, i_max, j_max, X, 1
    )

    # do the TFM with bays lambda
    pos, traction, traction_magnitude, f_n_m, _, _ = reg_fourier_tfm(
        Ftux, Ftuy, L, 1, s, meshsize, i_max, j_max, grid_mat, pix_per_mu, 0
    )

    # rescale traction with proper Young's modulus
    traction = E * traction
    traction_magnitude = E * traction_magnitude
    f_n_m = E * f_n_m

    # off with the shapes flip back into position
    traction_magnitude = traction_magnitude.reshape(i_max, j_max).T
    traction_magnitude = np.flip(traction_magnitude, axis=0)

    # Calculate inner product of traction (f_n_m or traction) and displacement (u) fields
    # Strain energy is the integral of resulting scalar field on defined domain

    return traction_magnitude, f_n_m, L
