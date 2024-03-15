# Import general libraries and modules
import os
import pickle
import tempfile
from typing import Tuple, Type, Union, Any
import numpy as np
import tifffile
from shapely import geometry

# Import custom modules from the 'pytraction' package
from pytraction.process import iterative_piv
from pytraction.preprocess import _find_uv_outside_single_polygon, normalize


def custom_noise(tiff_stack: np.ndarray,
                 config: dict
                 ) -> float:
    """
    Function to calculate custom noise value beta, representing the reciprocal of the variance of noise in a given image
    stack.

    @param  tiff_stack: Image stack in .tiff format
    @param  config: Configuration file for TFM Analysis
    """
    tmpdir = tempfile.gettempdir()  # Get system's temporary directory
    destination = f"{tmpdir}/tmp_noise.pickle"  # Define path to save cache file
    cache = dict()

    # Check if cache file with beta values for specific image stack already exists to save time
    if os.path.exists(destination):
        with open(destination, "rb") as f:
            cache = pickle.load(f)  # Load previous computed noise data
        beta = cache.get(tiff_stack, None)  # Check if there is a beta value for the current tiff_stack

        if beta:
            return beta  # Returns value immediately if present

    # Calculate beta value
    tiff_noise_stack = tifffile.imread(tiff_stack)  # Reads in images of (t,w,h) form
    un, vn = np.array([]), np.array([])  # Arrays to store components of displacement vectors

    # Calculate maximum between number of (time-frames - 1) and 2
    # This ensures that there are enough frames for beta calculations
    max_range = max(tiff_noise_stack.shape[0] - 1, 2)

    for i in range(max_range):  # Iterates at least over index 0 and 1
        # Select two subsequent images
        img = normalize(tiff_noise_stack[i, :, :])
        ref = normalize(tiff_noise_stack[i + 1, :, :])  # ToDo: Index out of boundary exception if less than 2?

        # Particle Image Velocimetry between img and ref
        # 'iterative_piv' from pytraction.process
        x, y, u, v, stack = iterative_piv(img, ref, config)  # Returns vectors and positions

        # Append u,v to un and vn arrays
        un = np.append(un, u)
        vn = np.append(vn, v)

    noise_vec = np.array([un.flatten(), vn.flatten()])  # Flatten displacement vectors and concatenate them
    var_noise = np.var(noise_vec)  # Calculate variance of noise vector
    beta = 1 / var_noise  # Reciprocal represents quality of PIV operation and is a measure of the inverse noise level
    cache[tiff_stack] = beta  # Save beta value with 'tiff_stack' as key

    # Save cache file to system's temporary directory
    with open(destination, "wb") as f:
        pickle.dump(cache, f)

    return beta  # ToDo: Call noise measurement in PIV function to avoid double calculation of displacement field


def get_noise(config,
              x: np.ndarray,
              y: np.ndarray,
              u: np.ndarray,
              v: np.ndarray,
              polygon: Union[Type[geometry.Polygon], None],
              custom_noise: Union[np.ndarray, None]
              ) -> float:
    """
    Function to calculate beta noise value based on input data.

    @param  x: x-position of deformation vector
    @param  y: y-position of deformation vector
    @param  u: u-component of deformation vector
    @param  v: v-component of deformation vector
    @param  polygon: shapely polygon to test which (x_i, y_i) is within
    @param  custom_noise: Image stack in .tiff format
    """

    # If ROI (polygon) is set, use vectors outside of polygon for noise calculation
    if polygon:
        noise_vec = _find_uv_outside_single_polygon(x=x, y=y, u=u, v=v, polygon=polygon)
    # If custom_noise is provided in form of a tiff_stack, calculate and return beta value
    elif custom_noise:
        return custom_noise(tiff_stack=custom_noise, config=config)
    # Else calculate beta value in small region of image
    else:
        noise = 10  # Constant for used image size
        xn, yn, un, vn = x[:noise], y[:noise], u[:noise], v[:noise]  # ToDo: xn, yn unused
        noise_vec = np.array([un.flatten(), vn.flatten()])  # Flatten displacement vectors and concatenate them

    var_noise = np.var(noise_vec)  # Calculate variance of noise vector
    beta = 1 / var_noise  # Reciprocal of displacement variance is a measure of the inverse noise level

    return beta
