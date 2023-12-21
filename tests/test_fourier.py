import numpy as np
from pytraction.fourier import fourier_xu, reg_fourier_tfm


def test__fourier_xu():
    # Define variables
    meshsize = 10
    E = 1000
    s = 0.5

    # Define example positional coordinates
    x, y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

    # Define vector components of example field
    u = -y / np.sqrt(x ** 2 + y ** 2)
    v = x / np.sqrt(x ** 2 + y ** 2)

    # Bring into correct form
    pos = np.array([x.flatten(), y.flatten()])
    vec = np.array([u.flatten(), v.flatten()])

    grid_mat, i_max, j_max, X, fuu, Ftux, Ftuy, u = fourier_xu(pos=pos, vec=vec, meshsize=meshsize, E=E, s=s, grid_mat=[])

    pass
