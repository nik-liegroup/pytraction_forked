import numpy as np
from shapely import geometry
from pytraction.noise import get_noise


def test_get_noise():
    x = np.linspace(0, 100, 200)
    x = np.repeat(x[:, np.newaxis], len(x), axis=1)

    y = np.linspace(0, 100, 200)
    y = np.repeat(y[:, np.newaxis], len(y), axis=1)

    u = np.random.rand(len(x), len(y))
    v = np.random.rand(len(x), len(y))

    coords = ((0., 0.), (0., 10.), (10., 10.), (10., 0.), (0., 0.))

    polygon = geometry.Polygon(coords)
    noise_poly = geometry.Polygon(coords)
    noise_array = np.array([u.flatten(), v.flatten()])

    beta1 = get_noise(x, y, u, v, None, None)
    beta2 = get_noise(x, y, u, v, polygon, None)
    beta3 = get_noise(x, y, u, v, None, noise_array)
    beta4 = get_noise(x, y, u, v, None, noise_poly)