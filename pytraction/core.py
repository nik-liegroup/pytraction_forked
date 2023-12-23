# Import general libraries and modules
import io
import os
import pickle
import tempfile
from typing import Tuple, Type, Union, Any
import h5py
import numpy as np
import segmentation_models_pytorch as smp
import tifffile
import torch
import yaml
from shapely import geometry

# Import custom modules from the 'pytraction' package
from pytraction.dataset import Dataset
from pytraction.net.dataloader import get_preprocessing
from pytraction.preprocess import (_create_crop_mask_targets, _get_min_window_size, _get_polygon_and_roi,
                                   _get_raw_frames, _load_frame_roi)
from pytraction.process import calculate_traction_map, iterative_piv
from pytraction.roi import roi_loaders
from pytraction.utils import normalize


class TractionForceConfig(object):
    """
    Configuration class for traction force microscopy analysis.
    Inherits from 'object' class (default).
    """
    def __init__(
            self,
            E: float,  # ToDo: Rename to improve understandability
            scaling_factor: float,
            config_path: str,
            min_window_size: Union[int, None] = None,
            meshsize: float = 10,
            s: float = 0.5,  # ToDo: Rename to improve understandability
            knn: bool = True,
            cnn: bool = True,
            **kwargs,
    ):
        """
        @param  E: Young's modulus of culture substrate in Pa
        @param  scaling_factor: Pixels per micrometer
        @param  config_path: System path to config.yaml file
        @param  min_window_size: Must be multiple of base 2 i.e. 8, 16, 32, 64. Determines the size of the subregions
        used for tracking particle motion which should be adjusted to bead density of the input images
        @param  meshsize: Specifies number of grid intervals to interpolate displacement field on
        @param  s: Poisson's ratio of substrate
        @param knn: Load K-nearest neighbors model to predict minimum window size based on bead density. Used only if
        "min_window_size" variable is not set
        @param cnn: Load convolutional neural network model for predicting the cell contour in an image
        @param **kwargs
        """
        # Load and configure parameters from a YAML file
        self.config = self._config_yaml(config_path=config_path, E=E, min_window_size=min_window_size, s=s,
                                        meshsize=meshsize, scaling_factor=scaling_factor)

        # Load K-nearest neighbors (KNN) model if enabled
        self.knn = self._get_knn_model() if knn else None

        # Load a Convolutional Neural Network (CNN) model if enabled
        # Get device from config.yaml file ("device" is a nested key of "settings" key)
        # This deep learning model architecture requires a preprocessing function
        self.model, self.pre_fn = (
            self._get_cnn_model(device=self.config["settings"]["device"]) if cnn else (None, None)
        )

        # Set additional parameters by iterating over key: value pairs
        for k, v in kwargs.items():
            self.config[k] = v

    def __repr__(self):
        """
        Custom representation of the object, when called by the built-in repr() function.
        """
        pass

    @staticmethod
    def _config_yaml(
            E: float,
            scaling_factor: float,
            config_path: str,
            min_window_size: Union[int, None],
            meshsize: float,
            s: float
    ) -> dict:
        """
        Import the config.yaml file from system path and parse to python dictionary.

        @param  E: Young's modulus of culture substrate in Pa
        @param  scaling_factor: Pixels per micrometer
        @param  config_path: System path to config.yaml file
        @param  min_window_size: Must be multiple of base 2 i.e. 8, 16, 32, 64. Initial data suggest a parameter between
        8 and 64 will be suitable for most applications but depends on the bead density of the input images
        @param  meshsize: Specifies number of grid intervals to interpolate displacement field on
        @param  s: Poisson's ratio of substrate
        """
        with open(config_path, "r") as config_file:
            config = yaml.load(stream=config_file, Loader=yaml.FullLoader)  # Parse yaml file to python dictionary

        # Overwrite parts of imported config dictionary with user input data
        config["tfm"]["E"] = E
        config["tfm"]["pix_per_mu"] = scaling_factor
        config["piv"]["min_window_size"] = (
            min_window_size
            if min_window_size is not None
            else config["piv"]["min_window_size"]
        )
        config["tfm"]["s"] = (
            s if s is not None else config["tfm"]["s"]
        )
        config["tfm"]["meshsize"] = (
            meshsize if meshsize is not None else config["tfm"]["meshsize"]
        )
        return config

    @staticmethod
    def _get_cnn_model(device: str) -> Tuple[Any, Any]:
        """
        Load a Convolutional Neural Network (CNN) model.

        @param  device: Ensures that CNN model can be used on the computing device (cpu, cuda, ...)
        """
        # New path to the model file in the package directory
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'cnn_model.pth')

        # Check if the model file exists at the new path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please make sure it's in the 'models' folder.")

        # Load the CNN model
        # "cpu" ensures that model can be used on a CPU even if the original training was done on a different device
        cnn_model = torch.load(f=model_path, map_location="cpu")  # ToDO: Rename to cnn_model

        if device == "cuda" and torch.cuda.is_available():
            best_model = cnn_model.to("cuda")

        # Retrieves a preprocessing function for images from the segmentation_models_pytorch library
        preproc_fn = smp.encoders.get_preprocessing_fn("efficientnet-b1", "imagenet")
        preprocessing_fn = get_preprocessing(preproc_fn)

        return cnn_model, preprocessing_fn

    @staticmethod
    def _get_knn_model() -> Any:
        """
        Load a K-nearest neighbors (KNN) model.
        """
        # New path to the KNN model file in the package directory
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'knn_model.pickle')

        # Check if the KNN model file exists at the new path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"KNN model file not found at {model_path}. Please make sure it's in the 'models' folder.")

        with open(model_path, "rb") as f:
            knn_model = pickle.load(f)

        return knn_model

    @staticmethod
    def load_data(
            img_path: str,
            ref_path: str,
            roi_path: str = ""
    ) -> Tuple[np.ndarray, np.ndarray, Union[Tuple[list, list], list, None]]:
        """
        Load image data, reference data, and ROI data from given paths.

        @param  img_path: System path to image .tiff file
        @param  ref_path: System path to reference .tiff file
        @param  roi_path: System path to file containing ROI information (.csv, .roi or .zip)
        """
        #  Read .tiff image file as numpy array in (t,c,w,h) format
        img = tifffile.imread(img_path)

        #  Read .tiff reference file as numpy array in (c,w,h) format
        ref = tifffile.imread(ref_path)

        roi = roi_loaders(roi_path)  # Load ROI file

        # Check if img & ref are instances of the 'np.ndarray' class
        if not isinstance(img, np.ndarray) or not isinstance(ref, np.ndarray):
            msg = f"Image data not loaded for {img_path} or {ref_path}"
            raise TypeError(msg)

        if len(img.shape) != 4:
            msg = f"Please ensure that the input image has shape (t,c,w,h) the current shape is {img.shape}"
            raise RuntimeWarning(msg)

        if len(ref.shape) == 4:
            ref = ref[-1:, :, :, :]  # Project to last time-step
            ref = np.reshape(ref, (ref.shape[1], ref.shape[2], ref.shape[3]))

        if len(ref.shape) != 3:
            msg = f"Please ensure that the input ref image has shape (c,w,h) the current shape is {ref.shape}"
            raise RuntimeWarning(msg)

        return img, ref, roi


def _find_uv_outside_single_polygon(
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        polygon: Type[geometry.Polygon]  # Todo: Remove comma?
) -> np.ndarray:  # Returns (un, vn) array with noisy u and v components
    """
    Function to find u and v deformation field components outside a single polygon.

    @param  x: x-position of deformation vector
    @param  y: y-position of deformation vector
    @param  u: u-component of deformation vector
    @param  v: v-component of deformation vector
    @param polygon: shapely polygon to test which (x_i, y_i) is within
    """
    # Create empty list
    noise = []

    # Flatten multi-dim. arrays to 1D array and combine them element-wise, e.g. [(*,*,*,*), (*,*,*,*), ...]
    for (x0, y0, u0, v0) in zip(x.flatten(), y.flatten(), u.flatten(), v.flatten()):
        p1 = geometry.Point([x0, y0])  # Creates shapely point for each tuple (x0, y0, u0, v0) at coordinates (x0, y0)
        if not p1.within(polygon):
            noise.append(np.array([u0, v0]))  # Add (u0, v0) to noise list if not in polygon
    return np.array(noise)


# ToDo: Move noise calculations in separate .py file
def _custom_noise(tiff_stack: np.ndarray,
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


def _get_noise(config,
               x: np.ndarray,
               y: np.ndarray,
               u: np.ndarray,
               v: np.ndarray,
               polygon: Type[geometry.Polygon],
               custom_noise: np.ndarray
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
        return _custom_noise(tiff_stack=custom_noise, config=config)
    # Else calculate beta value in small region of image
    else:
        noise = 10  # Constant for used image size
        xn, yn, un, vn = x[:noise], y[:noise], u[:noise], v[:noise]  # ToDo: xn, yn unused
        noise_vec = np.array([un.flatten(), vn.flatten()])  # Flatten displacement vectors and concatenate them

    var_noise = np.var(noise_vec)  # Calculate variance of noise vector
    beta = 1 / var_noise  # Reciprocal of displacement variance is a measure of the inverse noise level

    return beta


def _write_frame_results(
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
        vec: np.ndarray,
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
def _write_metadata_results(results: type(h5py.File),
                            config: dict) -> type(h5py.File):
    # Create metadata group with a placeholder dataset
    results["metadata"] = 0

    # Iterate through the PIV and TFM configuration parameters and store them as metadata
    for k, v in config["piv"].items():
        results["metadata"].attrs[k] = np.void(str(v).encode())

    for k, v in config["tfm"].items():
        results["metadata"].attrs[k] = np.void(str(v).encode())
    return results


def process_stack(
        img_stack: np.ndarray,
        ref_stack: np.ndarray,
        config: type(TractionForceConfig),
        roi: Union[Tuple[list, list], list, None],
        bead_channel: int = 1,
        cell_channel: int = 0,
        crop: bool = False,
        custom_noise: Union[np.ndarray, None] = None
) -> type(Dataset):
    """
    Central function to calculate PIV, traction map & save results to HDF5 file.

    @param  img_stack: Image stack
    @param  ref_stack: Reference image
    @param  config: TractionForceConfig class instance for pytraction analysis
    @param  bead_channel: Bead channel occurrence (0 or 1)
    @param  cell_channel: Cell channel occurrence (0 or 1)
    @param  roi: Roi data as returned from load_data function
    @param  crop: # Crop the image to the selected ROI with a border margin of 10%
    @param  custom_noise: Image stack used for noise calculations
    """
    # Check if config is instance of TractionForceConfig
    if not isinstance(config, TractionForceConfig):
        msg = f"Please verify that config is a instance of TractionForceConfig{type(config)}"
        raise TypeError(msg)

    # Determine the number of time-frames in the image stack
    n_frames = img_stack.shape[0]

    # Create an in-memory binary buffer for storing results without creating a physical file
    bytes_hdf5 = io.BytesIO()

    # Open an HDF5 file for storing large and complex data structures
    with h5py.File(bytes_hdf5, "w") as results:
        # Loop through each time-frame
        for frame in list(range(n_frames)):
            # Load image, reference, and cell image in 8bit format for the current frame
            img, ref, cell_img = _get_raw_frames(
                img_stack=img_stack, ref_stack=ref_stack, frame=frame, bead_channel=bead_channel,
                cell_channel=cell_channel
            )

            # Get the minimum window size for PIV
            min_window_size = _get_min_window_size(img, config)
            config.config["piv"]["min_window_size"] = min_window_size

            # Load ROI for the current frame
            roi_i = _load_frame_roi(roi=roi, frame=frame, nframes=n_frames)

            # Segment most central cell (or use ROI) to define polygon around cell. Returns (None, None) if otherwise.
            polygon, pts = _get_polygon_and_roi(cell_img=cell_img, roi=roi_i, config=config)

            # Crop targets if necessary
            img, ref, cell_img, mask = _create_crop_mask_targets(
                img, ref, cell_img, pts, crop, pad=50
            )

            # Perform PIV to calculate displacement vectors (u, v) for positions (x, y)
            x, y, u, v, (stack, dx, dy) = iterative_piv(img, ref, config)

            # Calculate noise value beta inside ROI, segmented cell or whole image
            beta = _get_noise(config, x, y, u, v, polygon, custom_noise=custom_noise)

            # Create arrays for position (pos) and displacement vectors (vec)
            pos = np.array([x.flatten(), y.flatten()])
            vec = np.array([u.flatten(), v.flatten()])

            # Compute traction map, force field, and L_optimal
            traction_map, f_n_m, strain_energy, l_optimal = calculate_traction_map(
                pos,
                vec,
                beta,
                config.config["tfm"]["meshsize"],
                config.config["tfm"]["s"],
                config.config["tfm"]["pix_per_mu"],
                config.config["tfm"]["E"],
            )

            # Write results for the current frame to the HDF5 file
            results = _write_frame_results(
                results,
                frame,
                traction_map,
                f_n_m,
                stack,
                cell_img,
                mask,
                beta,
                l_optimal,
                pos,
                vec,
            )

        # Write metadata to the results file
        _write_metadata_results(results, config.config)

        # To recover information in the future, use the following syntax:
        # h5py.File(results)['metadata'].attrs['img_path'].tobytes()

    # Return the results as a Dataset
    return Dataset(bytes_hdf5)
