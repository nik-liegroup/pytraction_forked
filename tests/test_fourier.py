import matplotlib.pyplot as plt
from pytraction.fourier import fourier_xu, reg_fourier_tfm
from pytraction.optimal_lambda import optimal_lambda
from pytraction.utils import interp_vec2grid
from pytraction.postprocess import *

# Define variables
meshsize = 5
E = 1000
s = 0.48
pix_per_mu = 1
beta = 0.1

def test__fourier():
    # Define vortex positional coordinates and their respective centres
    x, y = np.meshgrid(np.linspace(-100, 100, 50), np.linspace(-100, 100, 50))
    x0, y0 = 20, 20

    # Define vector components of 2-vortex field
    r1 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    r2 = np.sqrt((x + x0) ** 2 + (y + y0) ** 2)
    ux = - (y - y0) / np.exp(0.3 * r1) - (y + y0) / np.exp(0.3 * r2)
    uy = (x - x0) / np.exp(0.3 * r1) + (x + x0) / np.exp(0.3 * r2)

    # Bring into correct form
    pos = np.array([x.flatten(), y.flatten()])
    vec = np.array([ux.flatten(), uy.flatten()])

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
    d_xx, d_yy, theta = contraction_moments(xx, yy, txx, tyy, pix_per_mu)
    d_xx_ft, d_yy_ft, theta_ft = contraction_moments_ft(ftfx, ftfy, kxx, kyy, pix_per_mu)

    print(f"Strain energy: {energy}")
    print(f"Principal axes ratio: {d_xx / d_yy} and {d_xx_ft / d_yy_ft}")
    print(f"Net contractile moment : {d_xx + d_yy} and {d_xx_ft + d_yy_ft}")
    print(f"Angle of rotation: {theta} and {theta_ft}")

    # Plots
    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 4))

    # Quiver plot for the first vector field
    axs[0, 0].quiver(x, y, ux, uy, color='blue')
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
    axs[2, 0].quiver(xx, yy, txx, tyy, color='blue')
    axs[2, 0].set_title('Traction vector field')

    # Scalar plot for the fifth vector field
    traction_magnitude = f_magnitude.reshape(i_max, j_max).T
    traction_magnitude = np.flip(traction_magnitude, axis=0)

    c_x = traction_magnitude.shape[0] // 2
    c_y = traction_magnitude.shape[1] // 2

    norm = np.sqrt(np.sum(txx) ** 2 + np.sum(tyy) ** 2)
    sum_x = np.sum(txx) / norm
    sum_y = np.sum(tyy) / norm

    axs[2, 1].imshow(traction_magnitude, cmap='viridis')
    axs[2, 1].quiver([c_x, c_x], [c_y, c_y], [2 * np.cos(theta), 2 * np.cos(theta)],
                     [2 * np.sin(theta), 2 * np.sin(theta)],
                     color='red', scale=10)
    axs[2, 1].quiver([c_x, c_x], [c_y, c_y], [2 * sum_x, 2 * sum_x], [2 * sum_y, 2 * sum_y],
                     color='orange', scale=10)
    axs[2, 1].set_title('Traction scalar field')

    plt.show()
