import numpy as np
from scipy.sparse import spdiags

from pytraction.utils import interp_vec2grid


def fourier_xu(pos, vec, meshsize, E, s, grid_mat):
    """
    Transform displacement field to Fourier space. The fourier transform of a 2D vector field are two FT of the
    respective x- and y-scalar fields. Again merging these two scalar fields yields the corresponding vector field in
    fourier space. There, the length and orientation of each vector represents the amplitude and phase of a frequency
    component.

    @param  pos:
    @param  vec:
    @param  meshsize: Must be smaller or equal to 1
    @param  E: Elastic modulus in Pa
    @param  s: Parameter of Green's function
    @param  grid_mat:
    """
    # Transform the position values to the deformed state
    new_pos = pos + vec

    # Interpolate shifted vector field onto rectangular grid
    grid_mat, u, i_max, j_max = interp_vec2grid(new_pos, vec, meshsize, grid_mat)

    # ToDo: Shapes might be off here!
    # ToDo: Is (i_max - 1) / 2) supposed to be (i_max / 2 - 1) since it is even?

    # Calculate an array of representable spatial frequencies (Natural numbers) in a discrete system up to the Nyquist
    # frequency and scale it with 2pi/(i_max * meshsize) to get the corresponding wave vectors
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

    # Add singleton dimension at the beginning of the array (1, N) -> (1, 1, N)
    kx_vec = np.expand_dims(kx_vec, axis=0)
    ky_vec = np.expand_dims(ky_vec, axis=0)

    # Create 2D matrix kx & ky with dim(i_max, N) and dim(j_max, N) where each row contains the same values as the
    # original kx_vec & ky_vec arrays
    kx = np.tile(kx_vec.T, (1, j_max))  # Transpose as np.tile works along the second axis
    ky = np.tile(ky_vec, (i_max, 1))

    # Set zero frequency component (Offset) to establishes a reference point against which variations can be measured
    kx[0, 0] = 1
    ky[0, 0] = 1

    # Calculate the magnitude of the wave vector
    k = np.sqrt(kx ** 2 + ky ** 2)

    # Calculate solution to Boussinesq equation (Green's function) given a point traction
    conf = 2 * (1 + s) / (E * k ** 3)  # Normalization coefficient

    # Derive components of the Green's function matrix
    gf_xx = conf * ((1 - s) * k ** 2 + s * ky ** 2)
    gf_xy = conf * (-s * kx * ky)
    gf_yy = conf * ((1 - s) * k ** 2 + s * kx ** 2)

    # Remove self interaction terms in matrix components
    gf_xx[0, 0] = 0
    gf_xy[0, 0] = 0
    gf_yy[0, 0] = 0

    # Account for (conjugate) symmetries in Fourier-transformed space as FT of a real function is symmetric
    # ToDo: Check if this is the correct approach
    gf_xy[int(i_max // 2), :] = 0  # Set values in middle row to zero
    gf_xy[:, int(j_max // 2)] = 0  # Set values in middle column to zero

    # Reshape matrices from dim(i_max, j_max) to dim(1, i_max * j_max) by concatenating rows
    g1 = gf_xx.reshape(1, i_max * j_max)
    g2 = gf_yy.reshape(1, i_max * j_max)
    g3 = gf_xy.reshape(1, i_max * j_max)

    # Create zero filled matrix in the shape of g3
    g4 = np.zeros(g3.shape)

    # Concatenate and transposing g1 & g2 along first axis resulting in array of dim(i_max * j_max, 2) and flatten by
    # concatenating rows
    x1 = np.array([g1, g2]).T.flatten()
    x2 = np.array([g3, g4]).T.flatten()

    # Adds a new axis to get array with dim(i_max * j_max * 2, 1)
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

    # Remove all NaN values in the displacement field
    # ToDo: Issue14
    u = np.nan_to_num(u)

    # Calculate 2D Fourier transform of displacement field components
    ftux = np.fft.fft2(u[:, :, 0]).T  # FT of x component of u
    ftuy = np.fft.fft2(u[:, :, 1]).T  # FT of y component of u

    # Reshaped into column vectors
    fux1 = ftux.reshape(i_max * j_max, 1)
    fux2 = ftuy.reshape(i_max * j_max, 1)

    # Combine x- and y-Fourier columns into a single array with resulting dim(i_max * j_max, 2)
    fuu = np.array([fux1, fux2]).T.flatten()

    # Add additional dimension to array for further processing
    fuu = np.expand_dims(fuu, axis=1)

    return grid_mat, i_max, j_max, X, fuu, ftux, ftuy, u


def reg_fourier_tfm(
    Ftux,
    Ftuy,
    L,
    E,
    s,
    cluster_size,
    i_max,
    j_max,
    grid_mat=None,
    pix_durch_my=None,
    zdepth=None,
    slim=False,
):

    V = 2 * (1 + s) / E
    # shapes might be off here!
    # construct wave vectors
    kx_vec = (
        2
        * np.pi
        / i_max
        / cluster_size
        * np.concatenate([np.arange(0, (i_max - 1) / 2), -np.arange(i_max / 2, 0, -1)])
    )
    kx_vec = np.expand_dims(kx_vec, axis=0)
    ky_vec = (
        2
        * np.pi
        / j_max
        / cluster_size
        * np.concatenate([np.arange(0, (j_max - 1) / 2), -np.arange(j_max / 2, 0, -1)])
    )
    ky_vec = np.expand_dims(ky_vec, axis=0)

    kx = np.tile(kx_vec.T, (1, j_max))
    ky = np.tile(ky_vec, (i_max, 1))

    # We ignore DC component below and can therefore set k(1,1) =1
    kx[0, 0] = 1
    ky[0, 0] = 1

    if slim:  # Slim output. Calculate only traction forces for the case z=0
        Ginv_xx = (
            (kx ** 2 + ky ** 2) ** (-1 / 2)
            * V
            * (kx ** 2 * L + ky ** 2 * L + V ** 2) ** (-1)
            * (kx ** 2 * L + ky ** 2 * L + ((-1) + s) ** 2 * V ** 2) ** (-1)
            * (
                kx ** 4 * (L + (-1) * L * s)
                + kx ** 2
                * ((-1) * ky ** 2 * L * ((-2) + s) + (-1) * ((-1) + s) * V ** 2)
                + ky ** 2 * (ky ** 2 * L + ((-1) + s) ** 2 * V ** 2)
            )
        )
        Ginv_yy = (
            (kx ** 2 + ky ** 2) ** (-1 / 2)
            * V
            * (kx ** 2 * L + ky ** 2 * L + V ** 2) ** (-1)
            * (kx ** 2 * L + ky ** 2 * L + ((-1) + s) ** 2 * V ** 2) ** (-1)
            * (
                kx ** 4 * L
                + (-1) * ky ** 2 * ((-1) + s) * (ky ** 2 * L + V ** 2)
                + kx ** 2 * ((-1) * ky ** 2 * L * ((-2) + s) + ((-1) + s) ** 2 * V ** 2)
            )
        )
        Ginv_xy = (
            (-1)
            * kx
            * ky
            * (kx ** 2 + ky ** 2) ** (-1 / 2)
            * s
            * V
            * (kx ** 2 * L + ky ** 2 * L + V ** 2) ** (-1)
            * (kx ** 2 * L + ky ** 2 * L + ((-1) + s) * V ** 2)
            * (kx ** 2 * L + ky ** 2 * L + ((-1) + s) ** 2 * V ** 2) ** (-1)
        )

        Ginv_xx[0, 0] = 0
        Ginv_yy[0, 0] = 0
        Ginv_xy[0, 0] = 0

        Ginv_xy[int(i_max / 2), :] = 0
        Ginv_xy[:, int(j_max / 2)] = 0
        Ftfx = Ginv_xx * Ftux + Ginv_xy * Ftuy
        Ftfy = Ginv_xy * Ftux + Ginv_yy * Ftuy

        # simply set variables that we do not need to calculate here to 0
        f_pos = 0
        f_nm_2 = 0
        f_magnitude = 0
        f_n_m = 0

        return f_pos, f_nm_2, f_magnitude, f_n_m, Ftfx, Ftfy

    else:  # full output, calculate traction forces with z>=0
        z = zdepth / pix_durch_my
        X = i_max * cluster_size / 2
        Y = j_max * cluster_size / 2
        if z == 0:
            g0x = (
                np.pi ** (-1)
                * V
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
                * V
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
            g0x = (
                np.pi ** (-1)
                * V
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
                * V
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

        Ginv_xx = (
            np.exp(np.sqrt(kx ** 2 + ky ** 2) * z)
            * (kx ** 2 + ky ** 2) ** (-1 / 2)
            * V
            * (
                np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * (kx ** 2 + ky ** 2) * L
                + V ** 2
            )
            ** (-1)
            * (
                4 * ((-1) + s) * V ** 2 * ((-1) + s + np.sqrt(kx ** 2 + ky ** 2) * z)
                + (kx ** 2 + ky ** 2)
                * (4 * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * L + V ** 2 * z ** 2)
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
                + V ** 2
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
            * (kx ** 2 + ky ** 2) ** (-1 / 2)
            * V
            * (
                np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * (kx ** 2 + ky ** 2) * L
                + V ** 2
            )
            ** (-1)
            * (
                4 * ((-1) + s) * V ** 2 * ((-1) + s + np.sqrt(kx ** 2 + ky ** 2) * z)
                + (kx ** 2 + ky ** 2)
                * (4 * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * L + V ** 2 * z ** 2)
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
                + V ** 2
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
            * (kx ** 2 + ky ** 2) ** (-1 / 2)
            * V
            * (
                np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * (kx ** 2 + ky ** 2) * L
                + V ** 2
            )
            ** (-1)
            * (
                2
                * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z)
                * (kx ** 2 + ky ** 2)
                * L
                * (2 * s + np.sqrt(kx ** 2 + ky ** 2) * z)
                + V ** 2
                * (
                    4 * ((-1) + s) * s
                    + (-2) * np.sqrt(kx ** 2 + ky ** 2) * z
                    + 4 * np.sqrt(kx ** 2 + ky ** 2) * s * z
                    + (kx ** 2 + ky ** 2) * z ** 2
                )
            )
            * (
                4 * ((-1) + s) * V ** 2 * ((-1) + s + np.sqrt(kx ** 2 + ky ** 2) * z)
                + (kx ** 2 + ky ** 2)
                * (4 * np.exp(2 * np.sqrt(kx ** 2 + ky ** 2) * z) * L + V ** 2 * z ** 2)
            )
            ** (-1)
        )

        Ginv_xx[0, 0] = 1 / g0x
        Ginv_yy[0, 0] = 1 / g0y
        Ginv_xy[0, 0] = 0

        Ginv_xy[int(i_max // 2), :] = 0
        Ginv_xy[:, int(j_max // 2)] = 0

        Ftfx = Ginv_xx * Ftux + Ginv_xy * Ftuy
        Ftfy = Ginv_xy * Ftux + Ginv_yy * Ftuy

        f_n_m = np.zeros(Ftfx.shape + (2,))
        f_n_m[:, :, 0] = np.real(np.fft.ifft2(Ftfx))
        f_n_m[:, :, 1] = np.real(np.fft.ifft2(Ftfy))

        f_nm_2 = np.zeros((i_max * j_max, 2, 1))
        f_nm_2[:, 0] = f_n_m[:, :, 0].reshape(i_max * j_max, 1)
        f_nm_2[:, 1] = f_n_m[:, :, 1].reshape(i_max * j_max, 1)

        f_pos = np.zeros((i_max * j_max, 2, 1))
        f_pos[:, 0] = grid_mat[:, :, 0].reshape(i_max * j_max, 1)
        f_pos[:, 1] = grid_mat[:, :, 1].reshape(i_max * j_max, 1)

        f_magnitude = np.sqrt(f_nm_2[:, 0] ** 2 + f_nm_2[:, 1] ** 2)

    return f_pos, f_nm_2, f_magnitude, f_n_m, Ftfx, Ftfy
