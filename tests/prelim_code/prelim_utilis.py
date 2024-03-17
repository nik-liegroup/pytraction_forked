import numpy as np
from scipy.sparse import diags


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

    gamma_glob = diags(diagonals=diagonals,
                       offsets=(0, len(gamma_2), -len(gamma_3)),
                       shape=(2*len(gamma_1), 2*len(gamma_1)))

    return gamma_glob


# Center 2D array to remove padding
def center_padding(array2dim, x, y):
    # Determine the start and end indices to get the central part
    center_i = (np.shape(array2dim)[0]) // 2
    center_j = (np.shape(array2dim)[1]) // 2

    result = array2dim[center_i - len(x) // 2: center_i + len(x) // 2,
                       center_j - len(y) // 2: center_j + len(y) // 2]
    return result
