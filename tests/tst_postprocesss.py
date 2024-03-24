import os
import numpy as np
import matplotlib.pyplot as plt
from pytraction import TractionForceConfig, TractionForceDataset, process_stack
from pytraction.postprocess import strain_energy
from pytraction.noise import get_noise
from tests.prelim_code.tst_utilis import *


def tst__postprocess():
    analyse_folder = os.path.join('data', 'example4')
    pix_per_mu = 9.64

    data_path = os.path.join(analyse_folder, 'h5',
                             '2DTFM_10kPa_hPAAGel_2_3T3Fibroblasts_2023_12_14_TimePos_Series_Position 7.h5')

    try:
        dataset_loaded = TractionForceDataset(data_path)
        x, y = dataset_loaded[0]["pos"][0]
        u, v = dataset_loaded[0]["vec"][0]
        f_n_m = dataset_loaded[0]["force_field"][0]
    except FileNotFoundError:
        exit(f'An error occurred while opening {data_path}.')

    point_dens_x = int(np.sqrt(len(x)))
    point_dens_y = int(np.sqrt(len(y)))

    xx = x.reshape(point_dens_x, point_dens_y)
    yy = y.reshape(point_dens_x, point_dens_y)

    uxx = u.reshape(point_dens_x, point_dens_y)
    uyy = v.reshape(point_dens_x, point_dens_y)

    txx = f_n_m[:, :, 0].reshape(point_dens_x, point_dens_y)
    tyy = f_n_m[:, :, 1].reshape(point_dens_x, point_dens_y)

    energy = strain_energy(xx, yy, uxx, uyy, txx, tyy, pix_per_mu)
    print(energy)


tst__postprocess()
