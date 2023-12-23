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
    first_row_vector = xx[0, :].reshape(1, -1).flatten()
    first_column_vector = yy[:, 0].reshape(1, -1).flatten()

    # Integrate energy density over whole domain
    energy = 0.5 * simps(simps(energy_dens, first_column_vector), first_row_vector)

    return energy
