import matplotlib.pyplot as plt
from scipy.linalg import inv
from pytraction.fourier import fourier_xu, reg_fourier_tfm
from pytraction.optimal_lambda import optimal_lambda
from pytraction.utils import interp_vec2grid
from pytraction.postprocess import *
from tests.prelim_code.tst_utilis import *

# Define parameters
point_dens = 20
meshsize = 5
E = 1000
s = 0.48
pix_per_mu = 1
beta = 0.1
sigma = 15
coeff = (1 + s) / (np.pi * E)


# Define tri-pole vector field component functions
def G_x(x_p, y_p, x0, y0):
    gx = (np.exp(-((x_p + x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)) -
          np.exp(-((x_p - x0) ** 2 + (y_p - y0) ** 2) / (sigma ** 2)) +
          np.exp(-((x_p - x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)))
    return gx


def G_y(x_p, y_p, x0, y0):
    gy = (np.exp(-((x_p + x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)) -
          np.exp(-((x_p - x0) ** 2 + (y_p - y0) ** 2) / (sigma ** 2)) -
          np.exp(-((x_p - x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)))
    return gy


# Define forward kernel component functions to calculate displacement field
def Kxx(x, y, x_p, y_p):
    r = np.sqrt((x - x_p) ** 2 + (y - y_p) ** 2)
    if r != 0:
        kxx = coeff * ((1 - s) / r + s * x ** 2 / r ** 3)
    else:
        kxx = 0
    return kxx


def Kyy(x, y, x_p, y_p):
    r = np.sqrt((x - x_p) ** 2 + (y - y_p) ** 2)
    if r != 0:
        kyy = coeff * ((1 - s) / r + s * y ** 2 / r ** 3)
    else:
        kyy = 0
    return kyy


def Kxy(x, y, x_p, y_p):
    r = np.sqrt((x - x_p) ** 2 + (y - y_p) ** 2)
    if r != 0:
        kxy = coeff * (s * x * y / r ** 3)
    else:
        kxy = 0
    return kxy


# Define the integrands of the convolution component-wise
def ux_integrand(x, y, x_p, y_p, x0, y0):
    dux = Kxx(x, y, x_p, y_p) * G_x(x_p, y_p, x0, y0) + Kxy(x, y, x_p, y_p) * G_y(x_p, y_p, x0, y0)
    return dux


def uy_integrand(x, y, x_p, y_p, x0, y0):
    duy = Kxy(x, y, x_p, y_p) * G_x(x_p, y_p, x0, y0) + Kyy(x, y, x_p, y_p) * G_y(x_p, y_p, x0, y0)
    return duy


# Map traction field components to displacement field components by evaluating 2D convolution integrals
def antiderivative(x_val, y_val, x_p_val, y_p_val, x0, y0):
    max_i = len(x_val)
    max_j = len(y_val)
    ux = np.zeros((max_i, max_j))
    uy = np.zeros((max_i, max_j))

    for i, x in enumerate(x_val):
        for j, y in enumerate(y_val):
            for k in range(max_i - 1):
                for l in range(max_j - 1):
                    xp1, xp2 = x_p_val[k], x_p_val[k + 1]
                    yp1, yp2 = y_p_val[l], y_p_val[l + 1]

                    dxp = xp2 - xp1
                    dyp = yp2 - yp1

                    integrand_values_ux = [
                        ux_integrand(x, y, xp, yp, x0, y0)
                        for xp in (xp1, xp2)
                        for yp in (yp1, yp2)
                    ]

                    integrand_values_uy = [
                        uy_integrand(x, y, xp, yp, x0, y0)
                        for xp in (xp1, xp2)
                        for yp in (yp1, yp2)
                    ]

                    ux[i, j] += 0.25 * sum(integrand_values_ux) * dxp * dyp
                    uy[i, j] += 0.25 * sum(integrand_values_uy) * dxp * dyp

            print(f'Integral at {x}, {y} completed')
    return ux, uy


def forward_kernel(x_val, y_val, x_p_val, y_p_val, x0, y0):
    # Displacement field as solution to forward problem
    ux, uy = antiderivative(x_val, y_val, x_p_val, y_p_val, x0, y0)
    xx, yy = np.meshgrid(x_val, y_val)
    tx, ty = G_x(xx, yy, x0, y0), G_y(xx, yy, x0, y0)

    # Inverse calculation of traction field
    tx_inv = []
    ty_inv = []

    for index, value_x in np.ndenumerate(xx):
        i, j = index
        x_ij = xx[i, j]
        y_ij = yy[i, j]

        # Get forward kernel
        r = np.sqrt(x_ij ** 2 + y_ij ** 2)
        cff = (1 + s) / (np.pi * E * r ** 3)
        kxx = cff * (1 - s) * r ** 2 + s * x_ij ** 2
        kyy = cff * (1 - s) * r ** 2 + s * y_ij ** 2
        kxy = cff * s * x_ij * y_ij
        K = [[kxx, kxy], [kxy, kyy]]

        # Invert 2x2 matrix
        K_inv = inv(K)

        # Map displacement field components to traction field components
        tx_tmp = K_inv[0, 0] * ux[i, j] + K_inv[0, 1] * uy[i, j]
        ty_tmp = K_inv[1, 0] * ux[i, j] + K_inv[1, 1] * uy[i, j]

        # Append arrays
        tx_inv = np.append(tx_inv, tx_tmp)
        ty_inv = np.append(ty_inv, ty_tmp)

    tx_inv = np.reshape(tx_inv, (np.shape(tx)[0], np.shape(tx)[1]))
    ty_inv = np.reshape(ty_inv, (np.shape(ty)[0], np.shape(ty)[1]))

    # Calculate contraction moments and principal axes angle
    d_xx, d_yy, theta = contraction_moments(xx, yy, tx, ty, pix_per_mu)

    # Create plot of vector fields
    fig, axs = plt.subplots(2, 2)

    # Quiver plot for the first vector field
    # Remove small values to avoid division by 0
    tx_norm = np.where((tx < 0.01) & (tx > - 0.01), np.nan, tx)
    ty_norm = np.where((ty < 0.01) & (ty > - 0.01), np.nan, ty)

    # Calculate normalization coefficients
    t_norm = np.sqrt(tx_norm ** 2 + ty_norm ** 2)

    axs[0, 0].quiver(xx, yy, tx / t_norm, ty / t_norm, color='blue')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_title('Original traction field')

    # Quiver plot for the second vector field
    axs[0, 1].quiver(yy, xx, ux, uy, color='blue')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_title('Displacement field as forward solution')

    # Quiver plot for the third vector field
    axs[1, 0].quiver(x_val, y_val, tx_inv, ty_inv, color='green')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[1, 0].set_title('Recovered traction field (Matrix inversion)')

    # Scalar plot for the fourth field
    traction_magnitude = np.sqrt(tx ** 2 + ty ** 2)

    traction_magnitude = np.flip(traction_magnitude, axis=0)

    c_x = traction_magnitude.shape[0] // 2
    c_y = traction_magnitude.shape[1] // 2
    axs[1, 1].imshow(traction_magnitude, cmap='viridis')
    axs[1, 1].quiver([c_x, c_x], [c_y, c_y], [2 * np.cos(theta), 2 * np.cos(theta)],
                     [2 * np.sin(theta), 2 * np.sin(theta)],
                     color='red', scale=10)
    axs[1, 1].set_title('Recovered traction scalar field')

    # Show the plot
    plt.show()


def fourier_calc(x_val, y_val, vx, vy, v_norm):
    # Bring into correct form
    pos = np.array([x_val.flatten(), y_val.flatten()])
    vec = np.array([vx.flatten(), vy.flatten()])

    # Interpolate displacement field onto rectangular grid using meshsize
    grid_mat, u, i_max, j_max = interp_vec2grid(pos, vec, meshsize, [])

    # Transform displacement field to fourier space
    ftux, ftuy, kxx, kyy, i_max, j_max, X = fourier_xu(u, i_max, j_max, E, s, meshsize)

    # Calculate lambda from bayesian model
    L, evidencep, evidence_one = optimal_lambda(
        beta, ftux, ftuy, kxx, kyy, E, s, meshsize, i_max, j_max, X, 1
    )

    # Calculate traction field in fourier space and transform back to spatial domain
    f_pos, f_nm_2, f_magnitude, f_n_m, ftfx, ftfy = reg_fourier_tfm(
        ftux, ftuy, kxx, kyy, L, E, s, meshsize, i_max, j_max, pix_per_mu, 0, grid_mat
    )

    # Extract positional and vector coordinates
    xx = grid_mat[:, :, 0]
    yy = grid_mat[:, :, 1]
    uxx = u[:, :, 0]
    uyy = u[:, :, 1]
    txx = f_n_m[:, :, 0]
    tyy = f_n_m[:, :, 1]

    # Calculate strain energy
    energy = strain_energy(xx, yy, txx, tyy, uxx, uyy, pix_per_mu)

    # Calculate contraction moments and principal axes angle
    d_xx, d_yy, theta = contraction_moments(xx, yy, txx.T, tyy.T, pix_per_mu)
    d_xx_ft, d_yy_ft, theta_ft = contraction_moments_ft(ftfx, ftfy, kxx, kyy, pix_per_mu)

    print(f"Strain energy: {energy}")
    print(f"Principal axes ratio: {d_xx / d_yy} and {d_xx_ft / d_yy_ft}")
    print(f"Net contractile moment : {d_xx + d_yy} and {d_xx_ft + d_yy_ft}")
    print(f"Angle of rotation: {theta} and {theta_ft}")

    # Plots
    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 4))
    plt.subplots_adjust(hspace=0.27)

    # Quiver plot for the first vector field
    axs[0, 0].quiver(x_val, y_val, vx, vy, color='blue')
    axs[0, 0].set_title('Displacement field')

    # Quiver plot for the second vector field
    axs[0, 1].quiver(xx, yy, uxx, uyy, color='red')
    axs[0, 1].set_title('Interpolation of displacement field')

    # Quiver plot for the third vector field
    axs[1, 0].quiver(kxx, kyy, ftux / np.sqrt(ftux ** 2 + ftuy ** 2), ftuy / np.sqrt(ftux ** 2 + ftuy ** 2),
                     color='green')
    axs[1, 0].set_title('Fourier transform of displacement field')

    # Quiver plot for the fourth vector field
    axs[1, 1].quiver(kxx, kyy, ftfx, ftfy, color='green')
    axs[1, 1].set_title('Fourier transform of traction field')

    # Quiver plot for the fifth vector field
    txx = np.rot90(txx, 2)
    tyy = np.rot90(tyy, 2)
    axs[2, 0].quiver(xx, yy, txx, tyy, color='blue')
    axs[2, 0].set_title('Traction vector field')

    # Scalar plot
    traction_magnitude = f_magnitude.reshape(i_max, j_max).T
    traction_magnitude = np.flip(traction_magnitude, axis=0)

    # Find center of traction field
    c_x = traction_magnitude.shape[0] // 2
    c_y = traction_magnitude.shape[1] // 2

    axs[2, 1].imshow(traction_magnitude, cmap='viridis')
    axs[2, 1].quiver([c_x, c_x], [c_y, c_y], [2 * np.cos(theta + np.pi), 2 * np.cos(theta + np.pi)],
                     [2 * np.sin(theta + np.pi), 2 * np.sin(theta + np.pi)],
                     color='red', scale=10)
    axs[2, 1].set_title('Traction scalar field')

    plt.show()


def tst__fourier():
    x0, y0 = 30, 30  # Gaussian peak and vortex centres

    # Vector field dimensions
    x_min, x_max = -100, 100
    y_min, y_max = -100, 100

    # Create positional coordinates
    x_val = np.linspace(x_min, x_max, point_dens)
    y_val = np.linspace(y_min, y_max, point_dens)
    x_p_val = np.linspace(x_min, x_max, point_dens)
    y_p_val = np.linspace(y_min, y_max, point_dens)

    # Test tri-pole field
    #forward_kernel(x_val, y_val, x_p_val, y_p_val, x0, y0)

    # Test vortex field
    xx, yy = np.meshgrid(x_val, y_val)
    vx, vy, v_norm = vortex(xx, yy, x0, y0)
    fourier_calc(xx, yy, vx, vy, v_norm)

tst__fourier()
