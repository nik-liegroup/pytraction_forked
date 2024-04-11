import os
import numpy as np
import matplotlib.pyplot as plt
from pytraction import TractionForceConfig, TractionForceDataset, process_stack
from pytraction.process import calculate_traction_map
from pytraction.noise import get_noise
from tests.prelim_code.tst_utilis import *


def tst__fourier_method():
    analyse_folder = os.path.join('data', 'example4')
    E = 10000
    pix_per_mu = 9.64

    config_path = os.path.join(analyse_folder, 'config.yaml')

    img_path = os.path.join(analyse_folder,
                            '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_TimePos_Series_Position 7.tiff')
    ref_path = os.path.join(analyse_folder,
                            '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_Pos_Series_Reference_Position 7.tiff')
    roi_path = os.path.join(analyse_folder,
                            '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_TimePos_Series_Position 7.roi')

    data_path = os.path.join(analyse_folder, 'h5',
                             '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_TimePos_Series_Position 7.h5')
    png_path = os.path.join(analyse_folder, 'png',
                            '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_TimePos_Series_Position 7.png')

    traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu,
                                          window_size=64,
                                          config_path=config_path,
                                          segment=False)

    try:
        dataset_loaded = TractionForceDataset(data_path)
        x, y = dataset_loaded[0]["pos"][0]
        u, v = dataset_loaded[0]["vec"][0]
        beta = get_noise(traction_config, x, y, u, v, None, custom_noise=None)
    except FileNotFoundError:
        exit(f'An error occurred while opening {data_path}.')

    # Get position and displacement vectors for Fourier method
    pos_ft = np.array([x.flatten(), y.flatten()])
    vec_ft = np.array([u.flatten(), v.flatten()])

    point_dens = int(np.sqrt(len(x)))
    xx_ft = x.reshape(point_dens, point_dens)
    yy_ft = np.flipud(y.reshape(point_dens, point_dens))

    # Get position and displacement vectors for BEM method
    pos_bem = np.concatenate([x.reshape(point_dens ** 2, 1),
                              y.reshape(point_dens ** 2, 1)])
    vec_bem = np.concatenate([u.reshape(point_dens ** 2, 1),
                              v.reshape(point_dens ** 2, 1)])
    x_val, y_val = (np.linspace(- (xx_ft[0, -1] - xx_ft[0, 0]) / 2, (xx_ft[0, -1] - xx_ft[0, 0]) / 2, point_dens),
                    np.linspace(- (yy_ft[-1, 0] - yy_ft[0, 0]) / 2, (yy_ft[-1, 0] - yy_ft[0, 0]) / 2, point_dens))
    xx_bem, yy_bem = np.meshgrid(x_val, y_val)

    # Compute traction map, force field, and L_optimal
    traction_map_fourier, f_n_m, strain_energy, l_optimal, txx, tyy = calculate_traction_map(
        pos_ft,
        vec_ft,
        beta,
        traction_config.config["tfm"]["meshsize"],
        traction_config.config["tfm"]["s"],
        traction_config.config["tfm"]["pix_per_mu"],
        traction_config.config["tfm"]["E"],
    )

    # Calculate BEM matrix
    gamma_glob = traction_bem(xx_bem,
                              yy_bem,
                              'conv',
                              point_dens,
                              traction_config.config["tfm"]["s"],
                              traction_config.config["tfm"]["E"])

    bem_inverse_f = tikhonov(gamma_glob, vec_bem, 0.01)
    bem_inverse_fx = bem_inverse_f[:point_dens ** 2].reshape(point_dens, point_dens).T
    bem_inverse_fy = bem_inverse_f[point_dens ** 2:].reshape(point_dens, point_dens).T
    traction_map_bem = np.sqrt(bem_inverse_fx ** 2 + bem_inverse_fy ** 2)

    fig, ax = plt.subplots(1, 3, figsize=(30, 30))
    im1 = ax[0].imshow(traction_map_fourier,
                       extent=[np.min(xx_ft), np.max(xx_ft), np.min(yy_ft), np.max(yy_ft)],
                       interpolation="bicubic",
                       cmap="jet")
    ax[0].quiver(x, y, u, v, color='black')

    cbar = fig.colorbar(im1, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
    cbar.ax.tick_params(labelsize=14)

    im2 = ax[1].imshow(traction_map_bem,
                       extent=[np.min(xx_ft), np.max(xx_ft), np.min(yy_ft), np.max(yy_ft)],
                       interpolation="bicubic",
                       cmap="jet")
    ax[1].quiver(x, y, u, v, color='black')

    cbar = fig.colorbar(im2, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
    cbar.ax.tick_params(labelsize=14)

    im3 = ax[2].imshow(traction_map_bem/np.max(traction_map_bem) -
                       traction_map_fourier/np.max(traction_map_fourier),
                       extent=[np.min(xx_ft), np.max(xx_ft), np.min(yy_ft), np.max(yy_ft)],
                       interpolation="bicubic",
                       cmap="jet")

    cbar = fig.colorbar(im3, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show()


tst__fourier_method()
