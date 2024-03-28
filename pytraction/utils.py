import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from typing import Tuple, Type, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import linalg

from pytraction.tractionforcedataset import TractionForceDataset


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
    LU = linalg.splu(A.tocsc(), diag_pivot_thresh=0)  # sparse LU decomposition

    return LU.L.dot(sparse.diags(LU.U.diagonal() ** 0.5)).tocsr()


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


# Shape function for boundary element method
def pyramid2dim(xx, yy, width_x, width_y):
    return (np.maximum(0, width_x - np.abs(xx)) *
            np.maximum(0, width_y - np.abs(yy)))


def pyramid2dim_ft(kxx, kyy, width_x, width_y):
    return (width_x * np.sinc(kxx * width_x) *
            width_y * np.sinc(kyy * width_y)) ** 2


# Crate differential operator matrix
def diff_operator(ft_gxx, ft_gxy, ft_gyy):
    gamma_1 = ft_gxx.flatten()
    gamma_2 = ft_gxy.flatten()
    gamma_3 = gamma_2
    gamma_4 = ft_gyy.flatten()

    diagonals = np.array([np.concatenate([gamma_1, gamma_4]).flatten(), gamma_2, gamma_3])

    gamma_glob = sparse.diags(diagonals=diagonals,
                              offsets=(0, len(gamma_2), -len(gamma_3)),
                              shape=(2 * len(gamma_1), 2 * len(gamma_1)))

    return gamma_glob


# Center 2D array to remove padding
def center_padding(array2dim, x, y):
    # Determine the start and end indices to get the central part
    center_i = (np.shape(array2dim)[0]) // 2
    center_j = (np.shape(array2dim)[1]) // 2

    result = array2dim[center_i - len(x) // 2: center_i + len(x) // 2,
             center_j - len(y) // 2: center_j + len(y) // 2]
    return result
