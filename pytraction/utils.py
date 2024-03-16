import os
import sys
from typing import Tuple, Type, Union

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.sparse import linalg as splinalg

from pytraction.dataset import Dataset


def bead_density(img: np.ndarray) -> float:
    """
    Calculate bead density from image frame.
    """
    # Normalize image and enhance grayscale-contrast
    clahe_img = clahe(img)
    # Binarize image using threshold and normalize to values between [0, 1]
    norm = (
        cv2.adaptiveThreshold(
            clahe_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2
        )
        / 255
    )

    ones = len(norm[norm == 1])  # Calculate number of beads

    # Calculate total area of image and bead density
    area = img.shape[0] * img.shape[1]
    area_beads = ones / area

    return area_beads


def remove_boarder_from_aligned(aligned_img: np.ndarray, aligned_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops black borders from aligned image and returns it together with same size reference.
    """
    # Find the indices of non-zero values along rows and columns
    non_zero_rows = np.where(~np.all(aligned_img == 0, axis=1))[0]
    non_zero_cols = np.where(~np.all(aligned_img == 0, axis=0))[0]

    # Select only the subarray that excludes the bordering zero rows and columns
    borderless_img = aligned_img[non_zero_rows[0]:non_zero_rows[-1] + 1, non_zero_cols[0]:non_zero_cols[-1] + 1]
    borderless_ref = aligned_ref[non_zero_rows[0]:non_zero_rows[-1] + 1, non_zero_cols[0]:non_zero_cols[-1] + 1]

    return borderless_img, borderless_ref


def sparse_cholesky(A):
    """[summary]

    # The input matrix A must be a sparse symmetric positive-definite.

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    n = A.shape[0]
    LU = splinalg.splu(A.tocsc(), diag_pivot_thresh=0)  # sparse LU decomposition

    return LU.L.dot(sparse.diags(LU.U.diagonal() ** 0.5)).tocsr()


def interp_vec2grid(
        pos: np.ndarray,
        vec: np.ndarray,
        meshsize: float,
        grid_mat=np.array([])
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Interpolates the vector field onto a rectangular grid which will be constructed using the data dimensions or can be
    handed manually using the grid_mat variable.
    """
    if not grid_mat:  # If no grid is provided (No grid equals [])
        # Get boundary dimensions of vector field to calculate grid dimensions
        max_pos = [np.max(pos[0]), np.max(pos[1])]  # Calculate maximum values of positions along x- and y-axis
        min_pos = [np.min(pos[0]), np.min(pos[1])]  # Calculate minimum values of positions along x- and y-axis

        # Calculate size of vector field in each direction and divide by meshsize yielding the number of mesh intervals
        i_max = np.floor((max_pos[0] - min_pos[0]) / meshsize)   # np.floor rounds result to integer number
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
        # [(a, a), (a, b),      [a, a,       [a, b
        #  (b, a), (b, b)] ->    b, b]  and   a, b]

        # Merges 2D xx and 2D yy array into one 3D array by combining the respective coordinates into pairs
        grid_mat = np.stack([xx, yy], axis=2)
        # [[a, a], [a, b]
        #  [b, a], [b, b]]

        # Perform interpolation of displacement vectors at pos onto a grid defined by xx and yy
        u = griddata(pos.T, vec.T, (xx, yy), method="cubic")

    else:
        # Calculate number of intervals in x- and y-direction
        i_max = np.size(grid_mat[:, :, 0]) - 1
        j_max = np.size(grid_mat[:, :, 1]) - 1

        # Perform interpolation of displacement vectors at pos onto a grid defined by grid_mat
        u = griddata(pos.T, vec.T, (grid_mat[:, :, 0], grid_mat[:, :, 1]), method="cubic")

    # Remove all NaN values in the displacement field
    u = np.nan_to_num(u)

    return grid_mat, u, int(i_max), int(j_max)


def clahe(data: np.ndarray) -> np.ndarray:
    """

    """
    img = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)  # Convert COLOR_GRAY2BGR image to COLOR_BGR2LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert COLOR_BGR2LAB image to LAB color space
    l, a, b = cv2.split(lab)  # Separate image into its lightness (L) and color (A and B) components

    # Enhance image contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)  # Apply CLAHE to L component
    limg = cv2.merge((cl, a, b))  # Merge modified L channel with original A and B channels
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)[:, :, 0]  # Convert enhanced LAB image back to RGB color space but
    # return only the L channel, which is the grayscale-enhanced image.


def plot(
    log: type(Dataset),
    frames: Union[list, int] = 0,
    vmax: float = None,
    mask: bool = True,
    figsize: tuple = (16, 16),
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """[summary]

    Args:
        log (Type[Dataset]): [description]
        frame (int, optional): [description]. Defaults to 0.
        vmax (float, optional): [description]. Defaults to None.
        mask (bool, optional): [description]. Defaults to True.
        figsize (tuple, optional): [description]. Defaults to (16,16).

    Returns:
        Tuple[plt.figure.Figure,plt.axes.Axes]: [description]
    """

    if isinstance(frames, int):
        frames = np.asarray([frames], dtype=int)

    traction_map, tf_dim, cell_roi, pos, vec, L = [], [], [], [], [], []

    # Find the most common image dimension in the time-series
    for frame in frames:
        frame = int(frame)
        dim_tmp = log[frame]["traction_map"][0]
        tf_dim.append(len(dim_tmp[0]))
    comm_dim = max(set(tf_dim), key=tf_dim.count)

    # Iterate over time-series results
    for frame in frames:
        frame = int(frame)
        log_frame = log[frame]

        # Filter images with dimensions different from most commmon dimension
        if len(log_frame["traction_map"][0][0]) != comm_dim:
            continue

        traction_map.append(log_frame["traction_map"][0])
        cell_roi.append(log_frame["cell_roi"][0])
        pos.append(log_frame["pos"][0])
        vec.append(log_frame["vec"][0])
        L.append(log_frame["L"][0])

    # Calculate mean fields and values for time-series

    traction_map_sum = np.sum(traction_map, axis=0)

    cell_roi = cell_roi[0]
    x, y = pos[0]
    u_sum, v_sum = np.sum(vec, axis=0)
    L_mean = np.mean(L)
    vmax = np.max(traction_map_sum) if not vmax else vmax

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    im1 = ax[0].imshow(
        traction_map_sum,
        interpolation="bicubic",
        cmap="jet",
        extent=[x.min(), x.max(), y.min(), y.max()],
        vmin=0,
        vmax=vmax,
    )
    ax[0].quiver(x, y, u_sum, v_sum)

    if mask and log[0]["mask_roi"][0].shape:
        mask = log[0]["mask_roi"][0]
        mask = np.ma.masked_where(mask == 255, mask)
        ax[0].imshow(mask, cmap="jet", extent=[x.min(), x.max(), y.min(), y.max()])

    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    im2 = ax[1].imshow(cell_roi, cmap="gray", vmax=np.max(cell_roi))

    cbar = fig.colorbar(im1, cax=cax1)
    cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=20)
    cbar.ax.tick_params(labelsize=20)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()
    return fig, ax
