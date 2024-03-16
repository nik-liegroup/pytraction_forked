import os
import pickle
import tempfile
import numpy as np
import tifffile

from typing import Tuple, Type, Union, Any
from shapely import geometry
from pytraction.process import iterative_piv
from pytraction.preprocess import normalize


def get_noise(x: np.ndarray,
              y: np.ndarray,
              u: np.ndarray,
              v: np.ndarray,
              polygon: Union[Type[geometry.Polygon], None],
              noise: Union[np.ndarray, Type[geometry.Polygon], None]
              ) -> float:
    """
    Function to calculate beta noise value based on input data.
    @param  x: x-position of deformation vector.
    @param  y: y-position of deformation vector.
    @param  u: u-component of deformation vector.
    @param  v: v-component of deformation vector.
    @param  polygon: Shapely polygon to test which (x_i, y_i) is within geometry.
    @param  noise: Image stack in .tiff format or shapely polygon.
    """
    if type(noise) is np.ndarray:
        # Use noise stack for calculation of beta
        noise_vec = np.array([noise[:, 0].flatten(), noise[:, 1].flatten()])
    elif type(noise) is type(geometry.Polygon):
        # Use custom noise polygon for calculation of beta
        noise_vec = find_uv_from_polygon(x=x, y=y, u=u, v=v, polygon=noise, outside=False)
    elif polygon is not None:
        # Find vectors outside of polygon for noise calculations
        noise_vec = find_uv_from_polygon(x=x, y=y, u=u, v=v, polygon=polygon, outside=True)
    else:
        # Use vectors contained in a small stripe along the top image border for noise calculations
        noise_square = 10
        un, vn = u[:noise_square, :], v[:noise_square, :]
        noise_vec = np.array([un.flatten(), vn.flatten()])

    # Calculate variance of noise vector
    var_noise = np.var(noise_vec)

    # Reciprocal of displacement variance is a measure of inverse noise level
    beta = 1 / var_noise

    return beta


def find_uv_from_polygon(
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        polygon: Type[geometry.Polygon],
        outside: bool
) -> np.ndarray:
    """
    Find u and v deformation field components outside a single polygon.
    """
    # Create empty list to store vector components
    noise = []

    # Iterate over vector components with their respective positional coordinates
    for (x0, y0, u0, v0) in zip(x.flatten(), y.flatten(), u.flatten(), v.flatten()):
        # Creates shapely point at coordinates (x0, y0)
        point = geometry.Point([x0, y0])
        if (not point.within(polygon)) and outside:
            # Appends displacement vector if outside of polygon
            noise.append(np.array([u0, v0]))
        elif (point.within(polygon)) and (not outside):
            # Appends displacement vector if inside of polygon
            noise.append(np.array([u0, v0]))

    return np.array(noise)
