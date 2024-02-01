import numpy as np
from scipy.sparse import spdiags


def fourier_xu(
        u: np.ndarray,
        i_max: int,
        j_max: int,
        E: float,
        s: float,
        meshsize: float
):
    """
    Transform displacement field u to Fourier space. The fourier transform of a 2D vector field are two FT of
    the respective x- and y-scalar fields. Again merging these two scalar fields yields the corresponding vector field
    in fourier space. There, the length and orientation of each vector represents the amplitude and phase of a frequency
    component.
    Returns the differential operator (X) acting on Fourier-transformed displacement fields, 2D Fourier transform
    components of displacement field (ftux, ftuy) and the fourier transformed displacement field u.

    @param  u: Displacement field containing positions and vectors
    @param  i_max:
    @param  j_max:
    @param  E: Elastic modulus of substrate in Pa
    @param  s: Poisson's ratio of substrate
    @param  meshsize: Defines meshsize of rectangular grid to interpolate displacement field on.
    """
    # Calculate 2D Fourier transform of displacement field components
    ftux = np.fft.fft2(u[:, :, 0]).T  # FT of x component of u
    ftuy = np.fft.fft2(u[:, :, 1]).T # FT of y component of u

    # Calculate an array of representable spatial frequencies (Natural numbers) in a discrete system up to the Nyquist
    # frequency and scale it with 2pi/(i_max * meshsize) to get the corresponding wave vectors
    # ToDo: Consider re-arranging k-field to represent ascending values
    kx_vec = (
        2
        * np.pi
        / i_max
        / meshsize
        * np.concatenate([np.arange(0, (i_max - 1) / 2, 1), -np.arange(i_max / 2, 0, -1)])
    )

    ky_vec = (
            2
            * np.pi
            / j_max
            / meshsize
            * np.concatenate([np.arange(0, (j_max - 1) / 2, 1), -np.arange(j_max / 2, 0, -1)])
    )

    # Creates rectangular grid from every combination of provided kx and ky coordinates
    kxx, kyy = np.meshgrid(kx_vec, ky_vec, indexing='ij')  # kx and ky are both 2D matrices

    # Calculate the wave vectors magnitudes
    k = np.sqrt(kxx ** 2 + kyy ** 2)
    k[0, 0] = 1

    # Calculate fourier transform of Boussinesq solution (Green's function) given a point traction
    conf = 2 * (1 + s) / (E * k ** 3)  # Define coefficient

    # Derive components of the Green's function matrix
    gf_xx = conf * ((1 - s) * k ** 2 + s * kyy ** 2)
    gf_xy = conf * (s * kxx * kyy)
    gf_yy = conf * ((1 - s) * k ** 2 + s * kxx ** 2)

    # Set all zero frequency components in greens function to zero
    gf_xx[0, 0] = 0
    gf_xy[0, 0] = 0
    gf_yy[0, 0] = 0

    # ToDo: Check if this is the correct approach (Set gf_xy for lowest frequencies to zero)
    gf_xy[int(i_max // 2), :] = 0  # Set values in middle row to zero
    gf_xy[:, int(j_max // 2)] = 0  # Set values in middle column to zero

    # Reshape matrices from dim(i_max, j_max) to dim(1, i_max * j_max) by concatenating rows
    g1 = gf_xx.reshape(1, i_max * j_max)
    g2 = gf_yy.reshape(1, i_max * j_max)
    g3 = gf_xy.reshape(1, i_max * j_max)

    # Create zero filled matrix in the shape of g3
    g4 = np.zeros(g3.shape)

    # Concatenate and transposing g1 & g2 along first axis resulting in array of dim(i_max * j_max, 1) and flatten by
    # concatenating rows
    x1 = np.array([g1, g2]).T.flatten()
    x2 = np.array([g3, g4]).T.flatten()

    # Transpose and add dummy dimension to get array with dim(i_max * j_max * 2, 1)
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)

    # Eliminate the padding of zeros in x3 that was added during the construction of g4
    x3 = x2[1:]

    # Create a column vector (pad) containing a single element, which is 0
    pad = np.expand_dims(np.array([0]), axis=1)

    # Concatenate three arrays along the first axis
    data = np.array([np.concatenate([x3, pad]).T, x1.T, np.concatenate([pad, x3]).T])
    data = np.squeeze(data, axis=1)  # Removes the unnecessary singleton dimension introduced by np.expand_dims

    # Create 2D sparse matrix representing the differential operator acting on Fourier-transformed displacement fields
    X = spdiags(data, (-1, 0, 1), len(x1), len(x1))

    return ftux, ftuy, kxx, kyy, i_max, j_max, X


def reg_fourier_tfm(
    ftux: np.ndarray,
    ftuy: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    L: float,
    E: float,
    s: float,
    meshsize: float,
    i_max: int,
    j_max: int,
    scaling_factor: float = None,
    zdepth: float = 0,
    grid_mat: np.ndarray = [],
    slim: bool = False,
):
    """
    Maps the fourier transformed displacement field (ftux, ftux) via the Boussinesq Green's function to the respective
    traction field and transforms the result back into the spatial domain.

    @param  ftux: x-component of Fourier transformed displacement field
    @param  ftuy: y-component of Fourier transformed displacement field
    @param  kx: x-component wave vector 2D grid
    @param  ky: y-component wave vector 2D grid
    @param  L: Optimal lambda value
    @param  E: Young's modulus of culture substrate in Pa
    @param  s: Poisson's ratio of substrate
    @param  meshsize: Specifies number of grid intervals to interpolate displacement field on
    @param  i_max:
    @param  j_max:
    @param  scaling_factor: Pixels per micrometer
    @param  zdepth: Distance between gel surface and imaging plane (Must be a positive number)
    @param  grid_mat:
    @param  slim:
    """
    # Define coefficient
    v = 2 * (1 + s) / E
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1

    k_inv = k ** (-1)

    # Slim output for optimal_lambda call: Calculate only traction forces for the case z=0
    if slim:
        # Derive components of the Green's function matrix
        # ToDo: Check mathematical background
        Ginv_xx = (
            k_inv
            * v
            * (kx ** 2 * L + ky ** 2 * L + v ** 2) ** (-1)
            * (kx ** 2 * L + ky ** 2 * L + ((-1) + s) ** 2 * v ** 2) ** (-1)
            * (
                kx ** 4 * (L + (-1) * L * s)
                + kx ** 2
                * ((-1) * ky ** 2 * L * ((-2) + s) + (-1) * ((-1) + s) * v ** 2)
                + ky ** 2 * (ky ** 2 * L + ((-1) + s) ** 2 * v ** 2)
            )
        )

        Ginv_yy = (
            k_inv
            * v
            * (kx ** 2 * L + ky ** 2 * L + v ** 2) ** (-1)
            * (kx ** 2 * L + ky ** 2 * L + ((-1) + s) ** 2 * v ** 2) ** (-1)
            * (
                kx ** 4 * L
                + (-1) * ky ** 2 * ((-1) + s) * (ky ** 2 * L + v ** 2)
                + kx ** 2 * ((-1) * ky ** 2 * L * ((-2) + s) + ((-1) + s) ** 2 * v ** 2)
            )
        )

        Ginv_xy = (
            (-1)
            * kx
            * ky
            * k_inv
            * s
            * v
            * (kx ** 2 * L + ky ** 2 * L + v ** 2) ** (-1)
            * (kx ** 2 * L + ky ** 2 * L + ((-1) + s) * v ** 2)
            * (kx ** 2 * L + ky ** 2 * L + ((-1) + s) ** 2 * v ** 2) ** (-1)
        )

        # Set all zero frequency components in Green's function to zero
        Ginv_xx[0, 0] = 0
        Ginv_yy[0, 0] = 0
        Ginv_xy[0, 0] = 0

        # ToDo: Check if this is the correct approach (Set Ginv_xy for lowest frequencies to zero)
        Ginv_xy[int(i_max / 2), :] = 0
        Ginv_xy[:, int(j_max / 2)] = 0

        # Calculate convolution of displacement field and Green's function in Fourier space
        ftfx = Ginv_xx * ftux + Ginv_xy * ftuy
        ftfy = Ginv_xy * ftux + Ginv_yy * ftuy

        # Set unused variables to 0
        f_pos = 0
        f_nm_2 = 0
        f_magnitude = 0
        f_n_m = 0

        return f_pos, f_nm_2, f_magnitude, f_n_m, ftfx, ftfy

    # Full output: Calculate traction forces with z>=0
    else:
        # Get number of pixels in z-direction
        z = zdepth / scaling_factor  # ToDo: Assumes same scaling factor for z as for x,y?

        # Calculate center coordinates of x- and y-axis
        X = i_max * meshsize / 2
        Y = j_max * meshsize / 2

        if z == 0:
            # Derive normalization factors for the forward Green's function matrix
            g0x = (
                np.pi ** (-1)
                * v
                * (
                    (-1) * Y * np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2))
                    + Y * np.log(X + np.sqrt(X ** 2 + Y ** 2))
                    + ((-1) + s)
                    * X
                    * (
                        np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2))
                        + (-1) * np.log(Y + np.sqrt(X ** 2 + Y ** 2))
                    )
                )
            )

            g0y = (
                np.pi ** (-1)
                * v
                * (
                    ((-1) + s)
                    * Y
                    * (
                        np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2))
                        + (-1) * np.log(X + np.sqrt(X ** 2 + Y ** 2))
                    )
                    + X
                    * (
                        (-1) * np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2))
                        + np.log(Y + np.sqrt(X ** 2 + Y ** 2))
                    )
                )
            )

        else:
            # Derive normalization factors for the forward Green's function matrix
            g0x = (
                np.pi ** (-1)
                * v
                * (
                    ((-1) + 2 * s) * z * np.arctan(X ** (-1) * Y)
                    + (-2)
                    * z
                    * np.arctan(
                        X * Y * z ** (-1) * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                    )
                    + z
                    * np.arctan(
                        X ** (-1) * Y * z * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                    )
                    + (-2)
                    * s
                    * z
                    * np.arctan(
                        X ** (-1) * Y * z * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                    )
                    + (-1) * Y * np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                    + Y * np.log(X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                    + (-1) * X * np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                    + s * X * np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                    + (-1)
                    * ((-1) + s)
                    * X
                    * np.log(Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                )
            )

            g0y = (
                (-1)
                * np.pi ** (-1)
                * v
                * (
                    ((-1) + 2 * s) * z * np.arctan(X ** (-1) * Y)
                    + (3 + (-2) * s)
                    * z
                    * np.arctan(
                        X * Y * z ** (-1) * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                    )
                    + z
                    * np.arctan(
                        X ** (-1) * Y * z * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                    )
                    + (-2)
                    * s
                    * z
                    * np.arctan(
                        X ** (-1) * Y * z * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                    )
                    + Y * np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                    + (-1)
                    * s
                    * Y
                    * np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                    + ((-1) + s) * Y * np.log(X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                    + X * np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                    + (-1) * X * np.log(Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                )
            )

        # Derive components of the Green's function matrix
        Ginv_xx = (
            np.exp(np.sqrt(kx ** 2 + ky ** 2) * z)
            * k_inv
            * v
            * (
                np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * (kx ** 2 + ky ** 2) * L
                + v ** 2
            )
            ** (-1)
            * (
                4 * ((-1) + s) * v ** 2 * ((-1) + s + np.sqrt(kx ** 2 + ky ** 2) * z)
                + (kx ** 2 + ky ** 2)
                * (4 * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * L + v ** 2 * z ** 2)
            )
            ** (-1)
            * (
                (-2)
                * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z)
                * (kx ** 2 + ky ** 2)
                * L
                * (
                    (-2) * ky ** 2
                    + kx ** 2 * ((-2) + 2 * s + np.sqrt(kx ** 2 + ky ** 2) * z)
                )
                + v ** 2
                * (
                    kx ** 2
                    * (
                        4
                        + (-4) * s
                        + (-2) * np.sqrt(kx ** 2 + ky ** 2) * z
                        + ky ** 2 * z ** 2
                    )
                    + ky ** 2
                    * (
                        4
                        + 4 * ((-2) + s) * s
                        + (-4) * np.sqrt(kx ** 2 + ky ** 2) * z
                        + 4 * np.sqrt(kx ** 2 + ky ** 2) * s * z
                        + ky ** 2 * z ** 2
                    )
                )
            )
        )

        Ginv_yy = (
            np.exp(np.sqrt(kx ** 2 + ky ** 2) * z)
            * k_inv
            * v
            * (
                np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * (kx ** 2 + ky ** 2) * L
                + v ** 2
            )
            ** (-1)
            * (
                4 * ((-1) + s) * v ** 2 * ((-1) + s + np.sqrt(kx ** 2 + ky ** 2) * z)
                + (kx ** 2 + ky ** 2)
                * (4 * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * L + v ** 2 * z ** 2)
            )
            ** (-1)
            * (
                2
                * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z)
                * (kx ** 2 + ky ** 2)
                * L
                * (
                    2 * kx ** 2
                    + (-1) * ky ** 2 * ((-2) + 2 * s + np.sqrt(kx ** 2 + ky ** 2) * z)
                )
                + v ** 2
                * (
                    kx ** 4 * z ** 2
                    + (-2) * ky ** 2 * ((-2) + 2 * s + np.sqrt(kx ** 2 + ky ** 2) * z)
                    + kx ** 2
                    * (
                        4
                        + 4 * ((-2) + s) * s
                        + (-4) * np.sqrt(kx ** 2 + ky ** 2) * z
                        + 4 * np.sqrt(kx ** 2 + ky ** 2) * s * z
                        + ky ** 2 * z ** 2
                    )
                )
            )
        )

        Ginv_xy = (
            (-1)
            * np.exp(np.sqrt(kx ** 2 + ky ** 2) * z)
            * kx
            * ky
            * k_inv
            * v
            * (
                np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * (kx ** 2 + ky ** 2) * L
                + v ** 2
            )
            ** (-1)
            * (
                2
                * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z)
                * (kx ** 2 + ky ** 2)
                * L
                * (2 * s + np.sqrt(kx ** 2 + ky ** 2) * z)
                + v ** 2
                * (
                    4 * ((-1) + s) * s
                    + (-2) * np.sqrt(kx ** 2 + ky ** 2) * z
                    + 4 * np.sqrt(kx ** 2 + ky ** 2) * s * z
                    + (kx ** 2 + ky ** 2) * z ** 2
                )
            )
            * (
                4 * ((-1) + s) * v ** 2 * ((-1) + s + np.sqrt(kx ** 2 + ky ** 2) * z)
                + (kx ** 2 + ky ** 2)
                * (4 * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * L + v ** 2 * z ** 2)
            )
            ** (-1)
        )

        Ginv_xx[0, 0] = 1 / g0x
        Ginv_yy[0, 0] = 1 / g0y
        Ginv_xy[0, 0] = 0

        # ToDo: Check if this is the correct approach (Set Ginv_xy for lowest frequencies to zero)
        Ginv_xy[int(i_max // 2), :] = 0
        Ginv_xy[:, int(j_max // 2)] = 0

        # Calculate convolution of displacement field and Green's function in Fourier space
        ftfx = Ginv_xx * ftux + Ginv_xy * ftuy
        ftfy = Ginv_xy * ftux + Ginv_yy * ftuy

        # Avoid non-zero net force induced by spurious traction field
        ftfx[0, 0] = 0
        ftfy[0, 0] = 0

        # Initialize array of dim (i_max, i_max, 2)
        f_n_m = np.zeros(ftfx.shape + (2,))  # ToDo: Check what happens for i_max unequal i_max

        # Compute inverse discrete Fourier transform of traction field and extract real part of complex number
        f_n_m[:, :, 0] = np.real(np.fft.ifft2(ftfx))
        f_n_m[:, :, 1] = np.real(np.fft.ifft2(ftfy))

        f_nm_2 = np.zeros((i_max * j_max, 2, 1))
        f_nm_2[:, 0] = f_n_m[:, :, 0].reshape(i_max * j_max, 1)
        f_nm_2[:, 1] = f_n_m[:, :, 1].reshape(i_max * j_max, 1)

        f_pos = np.zeros((i_max * j_max, 2, 1))
        f_pos[:, 0] = grid_mat[:, :, 0].reshape(i_max * j_max, 1)
        f_pos[:, 1] = grid_mat[:, :, 1].reshape(i_max * j_max, 1)

        f_magnitude = np.sqrt(f_nm_2[:, 0] ** 2 + f_nm_2[:, 1] ** 2)

    return f_pos, f_nm_2, f_magnitude, f_n_m, ftfx, ftfy
