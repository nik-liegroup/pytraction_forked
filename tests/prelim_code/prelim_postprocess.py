import numpy as np
from scipy.integrate import simps, dblquad
from tests.prelim_code.prelim_utilis import *
from tests.prelim_code.tst_example_fields import *


def strain_energy(
        xx: np.ndarray,
        yy: np.ndarray,
        txx: np.ndarray,
        tyy: np.ndarray,
        uxx: np.ndarray,
        uyy: np.ndarray,
        pix_per_mu: float
):
    # Calculate pre-factor
    pre_factor = 1 / pix_per_mu

    # Flatten vectors to define integration intervals spaced accordingly to grid
    x = xx[0, :].reshape(1, -1).flatten()
    y = yy[:, 0].reshape(1, -1).flatten()

    # Calculate inner product of traction and displacement field
    energy_dens = txx * uxx + tyy * uyy

    # Integrate energy density over whole domain
    energy = 0.5 * simps(simps(energy_dens, y), x)

    # Scale to pico Joule (pNm) units
    energy = pre_factor ** 2 * energy

    return energy


def contraction_moments(
        xx: np.ndarray,
        yy: np.ndarray,
        txx: np.ndarray,
        tyy: np.ndarray,
        pix_per_mu: float
):
    # Calculate pre-factor
    pre_factor = 1 / pix_per_mu

    # Flatten vectors to define integration intervals spaced accordingly to grid
    x = xx[0, :].reshape(1, -1).flatten()
    y = yy[:, 0].reshape(1, -1).flatten()

    # Calculate components of contraction moment matrix in integral form
    m_xx = 0.5 * simps(simps(2 * xx * txx, x), y) * pre_factor ** 3
    m_yy = 0.5 * simps(simps(2 * yy * tyy, x), y) * pre_factor ** 3
    m_xy = 0.5 * simps(simps(xx * tyy + yy * txx, x), y) * pre_factor ** 3

    # Combine components into matrix form
    mat = np.array([[m_xx, m_xy], [m_xy, m_yy]])

    # Angle of rotation between coordinate system and principal axes of diagonalized matrix
    eigenvalues, eigenvectors = np.linalg.eig(mat)

    # If rot_angles[0] == rot_angles[1], then P is a 2x2 rotational matrix in P_inv @ mat @ P = dia
    rot_angles = np.arctan2(eigenvectors[1, :], eigenvectors[0, :])

    dia_xx, dia_yy = eigenvalues[0], eigenvalues[1]

    return dia_xx, dia_yy, rot_angles, 0.5 * np.arctan(2 * m_xy / (m_xx - m_yy))


# Define parameters
sigma = 5
x0, y0 = 5, 5

point_dens = 50
x_min, y_min = -10, -10
x_max, y_max = 10, 10

# Create grid of points in spatial coordinates
x_val, y_val = np.linspace(x_min, x_max, point_dens), np.linspace(y_min, y_max, point_dens)
xx, yy = np.meshgrid(x_val, y_val, indexing='xy')

txx, tyy, t_norm = vortex(xx, yy, x0, y0)
field_energy = strain_energy(xx, yy, txx, tyy, txx, txx, 1)

d_xx, d_yy, theta, theta2 = contraction_moments(xx, yy, txx, tyy, 1)

print(f'Strain energy: {field_energy}')
print(f'Moment d_xx: {d_xx}')
print(f'Moment d_xy: {d_yy}')
print(f'Theta: {np.rad2deg(theta)} and {np.rad2deg(theta2)}')
