from scipy.integrate import simps
import numpy as np


def strain_energy(
        xx: np.ndarray,
        yy: np.ndarray,
        txx: np.ndarray,
        tyy: np.ndarray,
        uxx: np.ndarray,
        uyy: np.ndarray
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
    energy = 0.5 * simps(simps(energy_dens, x), y)

    return energy


def contraction_moments(
        xx: np.ndarray,
        yy: np.ndarray,
        ftfx: np.ndarray,
        ftfy: np.ndarray,
        kxx: np.ndarray,
        kyy: np.ndarray,
        i: int = 2,
):
    """
    Calculate the contraction moments in linear approximation in Fourier space.
    """
    # Flatten vectors to define integration intervals spaced accordingly to grid
    x = xx[0, :].reshape(1, -1).flatten()
    y = yy[:, 0].reshape(1, -1).flatten()

    # ToDo: Check math!
    # Calculate components of contraction moment matrix
    m_xx = -(x/2) * (ftfx[1, i] + ftfx[1, i]) / (np.sqrt(kxx[1, i] ** 2 + kxx[1, i] ** 2))
    m_yy = -(y/2) * (ftfy[i, 1] + ftfy[i, 1]) / (np.sqrt(kyy[i, 1] ** 2 + kyy[i, 1] ** 2))
    m_xy = -(x/2) * (ftfy[1, i] + ftfx[i, 1]) / (np.sqrt(kxx[i, 1] ** 2 + kyy[1, i] ** 2))
    m_yx = -(y/2) * (ftfx[i, 1] + ftfy[1, i]) / (np.sqrt(kxx[i, 1] ** 2 + kyy[1, i] ** 2))

    return m_xx, m_yy, m_xy, m_yx
