import numpy as np
import h5py
from typing import Tuple, Type, Union, Any

from shapely import geometry
from pytraction.utils import bead_density


def _get_raw_frames(img_stack: np.ndarray, ref_stack: Union[np.ndarray, None], frame: int, bead_channel: int,
                    cell_channel: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normalized bead frame from image and reference stack and get cell channel from image stack.
    """
    img = normalize(np.array(img_stack[frame, bead_channel, :, :]))
    if ref_stack is not None:
        ref = normalize(np.array(ref_stack[bead_channel, :, :]))
    else:
        ref = normalize(np.array(img_stack[frame + 1, bead_channel, :, :]))
    cell_img = normalize(np.array(img_stack[frame, cell_channel, :, :]))
    return img, ref, cell_img


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Maps pixel intensity values to the interval [0, 255], corresponding to 8-bit images.
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))  # Spreads the range to [0, 1]
    return np.array(x * 255, dtype="uint8")  # Array is explicitly cast to unsigned 8-bit integer


def get_min_window_size(img: np.ndarray, config) -> int:
    """
    Calculate window size from bead density if min_window_size is not set in pyforce config file.
    """
    if not config.config["piv"]["min_window_size"]:
        density = bead_density(img)

        if config.knn is None:
            wnd_sz = config.config["piv"]["min_window_size"]
            msg = f"Window size set to {wnd_sz}, but KNN model was not loaded."
            raise RuntimeWarning(msg)

        # Use K-nearest neighbors (KNN) classifier to predict minimum window size based on bead density
        knn = config.knn
        min_window_size = knn.predict([[density]])
        print(f"Automatically selected window size of {min_window_size}")
        return int(min_window_size)

    else:
        return config.config["piv"]["min_window_size"]


def write_frame_results(
        results: type(h5py.File),
        frame: int,
        traction_map: np.ndarray,
        f_n_m: np.ndarray,
        stack: np.stack,
        cell_img: np.ndarray,
        mask: np.ndarray,
        beta: float,
        L_optimal: float,
        pos: np.ndarray,
        vec: np.ndarray
) -> type(h5py.File):
    """
    Function to write frame-specific results to an HDF5 file.

    @param  results: HDF5 file to write to
    @param  frame: Time-frame in image stack
    @param  traction_map: # 2D scalar map of traction stresses
    @param  f_n_m: 2D traction force vector field
    @param  stack: Combination of image and reference
    @param  cell_img: BF image of cell.
    @param  mask: Mask to separate cell from background
    @param  beta: Scalar value containing information about noise levels in PIV
    @param  L_optimal: # ToDo:Missing description
    @param  pos: Coordinates of deformation-vector positions
    @param vec: Coordinates of deformation-vectors
    """
    # Use variables to partly overwrite data in results file
    results[f"frame/{frame}"] = frame
    results[f"traction_map/{frame}"] = traction_map
    results[f"force_field/{frame}"] = f_n_m
    results[f"stack_bead_roi/{frame}"] = stack
    results[f"cell_roi/{frame}"] = cell_img
    results[f"mask_roi/{frame}"] = 0 if mask is None else mask
    results[f"beta/{frame}"] = beta
    results[f"L/{frame}"] = L_optimal
    results[f"pos/{frame}"] = pos
    results[f"vec/{frame}"] = vec
    return results


# Define a function to write metadata (to an HDF5 file?)
def write_metadata_results(results: type(h5py.File),
                           config: dict) -> type(h5py.File):
    # Create metadata group with a placeholder dataset
    results["metadata"] = 0

    # Iterate through the PIV and TFM configuration parameters and store them as metadata
    for k, v in config["piv"].items():
        results["metadata"].attrs[k] = np.void(str(v).encode())

    for k, v in config["tfm"].items():
        results["metadata"].attrs[k] = np.void(str(v).encode())
    return results
