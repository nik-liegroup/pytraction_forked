import numpy as np
from scipy.integrate import simps, dblquad
from tests.prelim_code.tst_utilis import *
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
    factor = pix_per_mu

    # Calculate inner product of traction and displacement field
    energy_dens = txx * uxx + tyy * uyy

    # Flatten vectors to define integration intervals spaced accordingly to grid
    x = xx[0, :].reshape(1, -1).flatten()
    y = yy[:, 0].reshape(1, -1).flatten()

    # Integrate energy density over whole domain
    energy = 0.5 * simps(simps(energy_dens, y), x)

    # Scale to pico Joule (pNm) units
    energy = energy / factor ** 3

    return energy


def contraction_moments(
        xx: np.ndarray,
        yy: np.ndarray,
        txx: np.ndarray,
        tyy: np.ndarray,
        pix_per_mu: float
):
    """
    Calculate the contraction moments in linear approximation in Fourier space.
    """
    # Flatten vectors to define integration intervals spaced accordingly to grid
    x = xx[0, :].reshape(1, -1).flatten()
    y = yy[:, 0].reshape(1, -1).flatten()

    # ToDo: Check math!
    # Calculate components of contraction moment matrix in integral form
    m_xx = 0.5 * simps(simps(xx * txx + xx * txx, x), y) * 10 ** (-6) / pix_per_mu ** 3
    m_yy = 0.5 * simps(simps(yy * tyy + yy * tyy, x), y) * 10 ** (-6) / pix_per_mu ** 3
    m_xy = 0.5 * simps(simps(xx * tyy + yy * txx, x), y) * 10 ** (-6) / pix_per_mu ** 3

    # Calculate absolute value of components and combine into matrix
    m_xx, m_yy, m_xy = abs(m_xx), abs(m_yy), abs(m_xy)

    # Angle of rotation between image coordinate system and principal axes of diagonalized matrix
    theta = 0.5 * np.arctan(2 * m_xy / (m_xx - m_yy))

    M = np.array([[m_xx, m_xy],
                  [m_xy, m_yy]])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    D = np.matmul(R.T, np.matmul(M, R))

    d_xx = D[0, 0]
    d_yy = D[1, 1]

    return d_xx, d_yy, theta



# Define the integrands for traction moments calculation
def traction_moments(xx, yy, txx, tyy):
    dm_xx = 2 * xx * txx
    dm_xy = xx * tyy + yy * txx
    dm_yy = 2 * yy * tyy

    m_xx = dblquad(dm_xx, -np.inf, np.inf, lambda y: -np.inf, lambda y: np.inf)
    m_xy = dblquad(dm_xy, -np.inf, np.inf, lambda y: -np.inf, lambda y: np.inf)
    m_yy = dblquad(dm_yy, -np.inf, np.inf, lambda y: -np.inf, lambda y: np.inf)

    return m_xx, m_xy, m_yy


# Define parameters
sigma = 5
x0, y0 = 5, 5

point_dens = 50
x_min, y_min = -10, -10
x_max, y_max = 10, 10

# Create grid of points in spatial coordinates
x_val, y_val = np.linspace(x_min, x_max, point_dens), np.linspace(y_min, y_max, point_dens)
xx, yy = np.meshgrid(x_val, y_val, indexing='xy')

txx, tyy, t_norm = tri_pole(xx, yy, x0, y0, sigma)
field_energy = strain_energy(xx, yy, txx, tyy, txx, txx, 1)

d_xx, d_yy, theta = contraction_moments(xx, yy, txx, tyy, 1)

print(f'Strain energy: {field_energy}')
print(f'Moment d_xx: {d_xx}')
print(f'Moment d_xy: {d_yy}')
print(f'Theta: {theta}')
