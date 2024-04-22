import os
import numpy as np
import pickle
import tempfile

from shapely.geometry import Polygon
from pytraction.core import (TractionForceConfig, process_stack)

from pytraction.tractionforcedataset import TractionForceDataset
from pytraction.inversion import traction_bem
from pytraction.regularization import tikhonov_simple


def tst_process_stack():
    elastic_modulus = 100
    poisson_ratio = 0.5

    vec_path = os.path.join("example_data", "example4_displacement", "example_displacement.npy")
    pos_path = os.path.join("example_data", "example4_displacement", "example_position.npy")
    vec, pos = np.load(vec_path), np.load(pos_path)

    gamma_glob = traction_bem(pos, method="conv", s=poisson_ratio, elastic_modulus=elastic_modulus)
    np.save("gamma_glob", gamma_glob)
    gamma_glob = np.load("gamma_glob.npy")

    fx, fy = tikhonov_simple(gamma_glob=gamma_glob, vec_u=vec, lambd=0.01)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.quiver(pos[:, :, 0], pos[:, :, 1], fx, fy, scale=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Vector Field')
    plt.grid(True)
    plt.show()

    print("Success")


