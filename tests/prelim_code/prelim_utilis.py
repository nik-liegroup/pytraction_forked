import numpy as np
from scipy.sparse import spdiags


# Shape function for boundary element method
def pyramid2dim(xx, yy, width_x, width_y):
    return (np.maximum(0, width_x - np.abs(xx)) *
            np.maximum(0, width_y - np.abs(yy)))


def pyramid2dim_ft(kxx, kyy, width_x, width_y):
    return (width_x * np.sinc(kxx * width_x) *
            width_y * np.sinc(kyy * width_y)) ** 2


# Crate differential operator matrix
def diff_operator(ft_gxx, ft_gxy, ft_gyy):
    # Get shape of block matrices
    i_max = (np.shape(ft_gxx)[0])
    j_max = (np.shape(ft_gyy)[1])

    g1 = ft_gxx.reshape(1, i_max * j_max)
    g2 = ft_gyy.reshape(1, i_max * j_max)
    g3 = ft_gxy.reshape(1, i_max * j_max)
    g4 = np.zeros(g3.shape)

    x1 = np.array([g1, g2]).T.flatten()
    x2 = np.array([g3, g4]).T.flatten()

    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)
    x3 = x2[1:]

    pad = np.expand_dims(np.array([0]), axis=1)

    data = np.array([np.concatenate([x3, pad]).T, x1.T, np.concatenate([pad, x3]).T])
    data = np.squeeze(data, axis=1)  # Removes the unnecessary singleton dimension introduced by np.expand_dims

    # Create 2D sparse matrix representing the Greens differential operator in Fourier space
    gamma_glob_2 = spdiags(data, (-1, 0, 1), len(x1), len(x1))


    # My code
    gamma_1 = np.diagflat(ft_gxx.reshape(1, i_max * j_max))
    gamma_2 = np.diagflat(ft_gxy.reshape(1, i_max * j_max))
    gamma_3 = gamma_2
    gamma_4 = np.diagflat(ft_gyy.reshape(1, i_max * j_max))

    gamma_glob = np.block([[gamma_1, gamma_2],
                           [gamma_3, gamma_4]])

    return gamma_glob


# Center 2D array to remove padding
def center_padding(array2dim, x, y):
    # Determine the start and end indices to get the central part
    center_i = (np.shape(array2dim)[0]) // 2
    center_j = (np.shape(array2dim)[1]) // 2

    result = array2dim[center_i - len(x) // 2: center_i + len(x) // 2,
                       center_j - len(y) // 2: center_j + len(y) // 2]
    return result
