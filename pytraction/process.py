import numpy as np
import cv2
from typing import Tuple
from openpiv import widim
from scipy.interpolate import griddata
from pytraction.optimal_lambda import optimal_lambda
from pytraction.utils import remove_boarder_from_aligned
from pytraction.regularization import *
from pytraction.inversion import traction_fourier, traction_bem
from pytraction.fourier import *


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


def interp_vec2grid(
        pos_flat: np.ndarray,
        vec_flat: np.ndarray,
        meshsize: float,
        pos_interp=np.array([])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates vector field onto rectangular grid which will be constructed using the meshsize parameter or can be
    handed manually using the grid_mat variable.
    """
    if not pos_interp:
        # Get boundary dimensions of vector field to calculate grid dimensions
        max_pos = [np.max(pos_flat[0]), np.max(pos_flat[1])]  # Calculate maximum value of position along x- and y-axis
        min_pos = [np.min(pos_flat[0]), np.min(pos_flat[1])]  # Calculate minimum value of position along x- and y-axis

        # Calculate size of vector field in each direction and divide by meshsize yielding the number of mesh intervals
        i_max = np.floor((max_pos[0] - min_pos[0]) / meshsize)  # np.floor rounds result to integer number
        j_max = np.floor((max_pos[1] - min_pos[1]) / meshsize)

        # Ensure that number of mesh intervals are even by subtracting the remainder when divided by 2
        i_max = i_max - np.mod(i_max, 2)
        j_max = j_max - np.mod(j_max, 2)

        # Generate evenly spaced grid points within the calculated dimensions of the grid by adding multiples of
        # meshsize to min x-y-values
        x_grid = min_pos[0] + np.arange(0.5, i_max, 1) * meshsize
        y_grid = min_pos[1] + np.arange(0.5, j_max, 1) * meshsize

        # Creates rectangular grid from every combination of provided x and y coordinates
        xx, yy = np.meshgrid(x_grid, y_grid)  # xx and yy are both 2D matrices

        # Merges 2D xx and 2D yy array along second axis
        pos_interp = np.stack([xx, yy], axis=2)

    # Perform interpolation of displacement vectors at pos onto a grid defined by grid_mat
    vec_interp = griddata(pos_flat.T, vec_flat.T, (pos_interp[:, :, 0], pos_interp[:, :, 1]), method="cubic")

    # Remove all NaN values in the displacement field
    vec_interp = np.nan_to_num(vec_interp)

    return pos_interp, vec_interp


def calculate_traction_map(pos: np.array,
                           vec_u: np.array,
                           beta: float,
                           s: float,
                           pix_per_mu: float,
                           E: float,
                           method: str = 'FT'):
    """
    Calculates 2D traction map given the displacement vector field and the noise value beta using an FFT or FEM
    (Boundary element method) approach.
    """
    # Predict lambda for tikhonov regularization from bayesian model
    # ToDo: Upgrade to new function in pytraction.regularization
    xx = pos[:, :, 0]
    x_val = xx[0, :]
    meshsize = x_val[1] - x_val[0]
    ft_ux_old, ft_uy_old, kxx_old, kyy_old, gamma_glob_old = fourier_xu(vec_u, E, s, meshsize)
    lamd, evidence, evidence_one = optimal_lambda(
        beta, ft_ux_old, ft_uy_old, kxx_old, kyy_old, E, s, meshsize, gamma_glob_old
    )
    # ToDo: END

    if method == 'FT':
        # Calculate simple inverse solution for regularization parameter estimation
        _, _, _, _, _, _, _, _, gamma_glob = traction_fourier(pos=pos,
                                                              vec=vec_u,
                                                              s=s,
                                                              elastic_modulus=E,
                                                              lambd=None,
                                                              scaling_factor=pix_per_mu,
                                                              zdepth=0)

        # Calculate traction field in fourier space and transform back to spatial domain
        fx, fy, _, _, _, _, _, _, _ = traction_fourier(pos=pos,
                                                       vec=vec_u,
                                                       s=s,
                                                       elastic_modulus=E,
                                                       lambd=lamd,
                                                       scaling_factor=pix_per_mu,
                                                       zdepth=0)

    elif method == 'BEM':
        gamma_glob = traction_bem(pos=pos, method='conv', s=s, elastic_modulus=E)
        fx, fy = tikhonov_simple(gamma_glob=gamma_glob, vec_u=vec_u, lambd=lamd)

    else:
        msg = ('Only fourier transform (FT) and boundary element method (BEM) are currently implemented to solve \
         inverse problem.')
        raise RuntimeError(msg)

    vec_f = np.stack((fx, fy), axis=2)

    return vec_f, lamd, evidence_one
