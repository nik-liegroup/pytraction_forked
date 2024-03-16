import numpy as np

from typing import Tuple
import cv2
from openpiv import widim
from pytraction.fourier import fourier_xu, reg_fourier_tfm
from pytraction.optimal_lambda import optimal_lambda
from pytraction.utils import remove_boarder_from_aligned, interp_vec2grid


def iterative_piv(img: np.ndarray, ref: np.ndarray, config):
    """
    Perform iterative PIV on drift corrected images and returns drift corrections (dx,dy), displacement vectors (u,v)
    for positions (x,y) and image stack.
    """
    # Check if displacements are too small, which would result in an iterative decrease in window size
    if np.allclose(img, ref, rtol=1e-05, atol=1e-08):
        msg = "Image and reference frame are approximately equal (Tolerance: 1e-05)."
        raise RuntimeError(msg)

    # Calculate drift in x,y and return drift corrected img
    dx, dy, img = align_slice(img, ref)
    if config.config["settings"]["crop_aligned_slice"]:
        # Crop images to remove black borders
        img, ref = remove_boarder_from_aligned(img, ref)

    # Calculate displacement field
    x, y, u, v, mask = compute_piv(img, ref, config)

    # Create drift corrected stack
    drift_corrected_stack = np.stack([img, ref])

    return x, y, u, v, dx, dy, drift_corrected_stack


def align_slice(img: np.ndarray, ref: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Given a bead and reference image, compute the drift using cv2.matchTemplate and return the drift corrected bead
    image along with the x drift (dx) and y drift (dy). The dx, dy shift is a measure of how much the image has moved
    with respect to the reference frame.
    """
    # Set window size in reference frame used for drift correction
    depth = int(min(img.shape) * 0.1)

    # Slide reference image (w x h) over input image (W x H) and calculate correlation landscape (W-w+1, H-h+1)
    # Assumes no displacements to be present in top left corner
    ccorr_normed = cv2.matchTemplate(
        img, ref[depth:-depth, depth:-depth], cv2.TM_CCORR_NORMED
    )

    # Flatten array and find index of maximal correlation
    ccorr_max_id = np.argmax(ccorr_normed, axis=None)

    # Convert flattened index back into a tuple of coordinates, describing top left corner of target rectangle
    max_ccorr = np.unravel_index(
        ccorr_max_id, ccorr_normed.shape
    )

    # Transform location of best match from correlation landscape to coordinate system of reference image
    # CC landscape is depth smaller in both x and y dimension
    dy = depth - max_ccorr[0]
    dx = depth - max_ccorr[1]
    rows, cols = img.shape

    # Initialize 2x3 transformation matrix
    # [[a, b, tx],
    # [c, d, ty]] -> a,d = 1: no scaling in x,y ; b,c = 0: no shearing or rotation ; tx, ty: translations
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply transformation to input image
    img = cv2.warpAffine(img, matrix, (cols, rows))

    return dx, dy, img


def compute_piv(img: np.ndarray,
                ref: np.ndarray,
                config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute deformation field using particle image velocimetry (PIV) with an implementation of window size coarsening.
    """
    # Select whole image for PIV
    mask = np.ones_like(ref).astype(np.int32)
    try:
        # Compute PIV using a window displacement iterative method
        x, y, u, v, mask = widim.WiDIM(  # Returns displacement vectors (u,v) for positions (x,y)
            ref.astype(np.int32),
            img.astype(np.int32),
            mask,
            **config.config["piv"],  # Unpack and pass the contents of a dictionary as keyword arguments
        )
        return x, y, u, v, mask
    except Exception as e:
        if isinstance(e, ZeroDivisionError):
            # Reduce window size and call compute_piv again
            config_tmp = config.copy()
            config_tmp.config["piv"]["min_window_size"] = (
                config.config["piv"]["min_window_size"] // 2
            )
            print(
                f"Reduced min window size to {config_tmp.config['piv']['min_window_size'] // 2} in recursive call"
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
    # ToDO: Check if reference frame update vec = vec + pos is necessary
    grid_mat, u, i_max, j_max = interp_vec2grid(pos, vec, meshsize, [])

    # Transform displacement field to fourier space
    ftux, ftuy, kxx, kyy, i_max, j_max, X = fourier_xu(u, i_max, j_max, E, s, meshsize)

    # Calculate lambda from bayesian model
    L, evidencep, evidence_one = optimal_lambda(
        beta, ftux, ftuy, kxx, kyy, E, s, meshsize, i_max, j_max, X
    )

    # Calculate traction field in fourier space and transform back to spatial domain
    f_pos, f_nm_2, f_magnitude, f_n_m, ftfx, ftfy = reg_fourier_tfm(
        ftux, ftuy, kxx, kyy, L, E, s, meshsize, i_max, j_max, pix_per_mu, 0, grid_mat
    )

    # Flip shapes back into position
    traction_magnitude = f_magnitude.reshape(i_max, j_max).T
    traction_magnitude = np.flip(traction_magnitude, axis=0)

    # Extract positional and vector coordinates
    xx = grid_mat[:, :, 0]
    yy = grid_mat[:, :, 1]
    uxx = u[:, :, 0]
    uyy = u[:, :, 1]
    txx = f_n_m[:, :, 0]
    tyy = f_n_m[:, :, 1]

    return traction_magnitude, f_n_m, L, evidence_one
