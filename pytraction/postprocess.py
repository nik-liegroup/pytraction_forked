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


def contraction_moments(
        ftfx: np.ndarray,
        ftfy: np.ndarray,
        kxx: np.ndarray,
        kyy: np.ndarray
):
    """
    Calculate the contraction moments in linear approximation in Fourier space.
    """
    # Flatten second respective row and column of wave vectors to avoid value 1 in [0, 0]
    kx = kxx[:, 1].reshape(1, -1).flatten()
    ky = kyy[1, :].reshape(1, -1).flatten()

    # Get index of first non-zero element in wave vectors
    ind_kx = (kx != 0).argmax(axis=0)
    ind_ky = (ky != 0).argmax(axis=0)

    # ToDo: Check math!
    # Calculate components of contraction moment matrix
    m_xx = (- 0.5 * complex(0, 1)
            * (ftfx[1, ind_kx] + ftfx[1, ind_kx]) / (np.sqrt(kxx[1, ind_kx] ** 2 + kxx[1, ind_kx] ** 2)))
    m_yy = (- 0.5 * complex(0, 1)
            * (ftfy[ind_ky, 1] + ftfy[ind_ky, 1]) / (np.sqrt(kyy[ind_ky, 1] ** 2 + kyy[ind_ky, 1] ** 2)))
    m_xy = (- 0.5 * complex(0, 1)
            * (ftfy[ind_kx, 1] + ftfx[1, ind_ky]) / (np.sqrt(kxx[1, ind_ky] ** 2 + kyy[ind_kx, 1] ** 2)))
    m_yx = (- 0.5 * complex(0, 1)
            * (ftfx[1, ind_ky] + ftfy[ind_kx, 1]) / (np.sqrt(kxx[1, ind_ky] ** 2 + kyy[ind_kx, 1] ** 2)))

    # Calculate absolute value of components and combine into matrix
    m_xx, m_yy, m_xy, m_yx = abs(m_xx), abs(m_yy), abs(m_xy), abs(m_yx)

    # Angle of rotation between image coordinate system and principal axes of diagonalized matrix
    theta = 0.5 * np.arctan(2 * m_xy / (m_xx - m_yy))

    M = np.array([[m_xx, m_xy],
                  [m_yx, m_yy]])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    D = np.matmul(R.T, np.matmul(M, R))

    d_xx = D[0, 0]
    d_yy = D[1, 1]

    return d_xx, d_yy, theta
