from scipy.integrate import simps
import numpy as np


def strain_energy(
        xx: np.ndarray,
        yy: np.ndarray,
        txx: np.ndarray,
        tyy: np.ndarray,
        uxx: np.ndarray,
        uyy: np.ndarray,
        pix_per_mu: float
):
    """
    Calculates strain energy of the displacement and traction field in the spatial domain.
    """
    # Calculate inner product of traction and displacement field
    energy_dens = txx * uxx.T + tyy * uyy.T

    # Flatten vectors to define integration intervals spaced accordingly to grid
    x = xx[0, :].reshape(1, -1).flatten()
    y = yy[:, 0].reshape(1, -1).flatten()

    # Integrate energy density over whole domain
    energy = 0.5 * simps(simps(energy_dens, y), x)

    # Scale to pico Joule (pNm) units
    energy = energy * 10 ** (-6) / pix_per_mu ** 3

    return energy


def contraction_moments_ft(
        ftfx: np.ndarray,
        ftfy: np.ndarray,
        kxx: np.ndarray,
        kyy: np.ndarray,
        pix_per_mu: float
):
    """
    Calculate the contraction moments in linear approximation in Fourier space.
    """
    # Flatten respective row and column of wave vectors
    kxx = kxx.T
    kyy = kyy.T

    kx = kxx[0, :].reshape(1, -1).flatten()
    ky = kyy[:, 0].reshape(1, -1).flatten()

    # Get index of first non-zero element in wave vectors
    ind_kx = (kx != 0).argmax(axis=0)
    ind_ky = (ky != 0).argmax(axis=0)

    # ToDo: Check math!
    # Calculate components of contraction moment matrix in Fourier space
    m_xx = (- 0.5 * complex(0, 1) * 10 ** (-6) / pix_per_mu ** 3
            * (ftfx[0, ind_kx] + ftfx[0, ind_kx]) / (np.sqrt(kxx[0, ind_kx] ** 2 + kxx[0, ind_kx] ** 2)))
    m_yy = (- 0.5 * complex(0, 1) * 10 ** (-6) / pix_per_mu ** 3
            * (ftfy[ind_ky, 0] + ftfy[ind_ky, 0]) / (np.sqrt(kyy[ind_ky, 0] ** 2 + kyy[ind_ky, 0] ** 2)))
    m_xy = (- 0.5 * complex(0, 1) * 10 ** (-6) / pix_per_mu ** 3
            * (ftfy[ind_kx, 0] + ftfx[0, ind_ky]) / (np.sqrt(kxx[0, ind_ky] ** 2 + kyy[ind_kx, 0] ** 2)))

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
    m_xx = 0.5 * simps(simps(xx * txx.T + xx * txx.T, x), y) * 10 ** (-6) / pix_per_mu ** 3
    m_yy = 0.5 * simps(simps(yy * tyy.T + yy * tyy.T, x), y) * 10 ** (-6) / pix_per_mu ** 3
    m_xy = 0.5 * simps(simps(xx * tyy.T + yy * txx.T, x), y) * 10 ** (-6) / pix_per_mu ** 3

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
