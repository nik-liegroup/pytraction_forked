# Import general libraries and modules
import io
import os
import pickle
from typing import Tuple, Type, Union, Any
import h5py
import numpy as np
import segmentation_models_pytorch as smp
import tifffile
import torch
import yaml

# Import custom modules from the 'pytraction' package
from pytraction.dataset import Dataset
from pytraction.net.dataloader import get_preprocessing
from pytraction.preprocess import (create_crop_mask_targets, get_min_window_size,
                                   get_polygon_and_roi, _get_raw_frames,
                                   load_frame_roi, write_frame_results,
                                   write_metadata_results)
from pytraction.process import calculate_traction_map, iterative_piv
from pytraction.roi import roi_loaders
from pytraction.noise import get_noise


class TractionForceConfig(object):
    """
    Configuration class for 2D traction force microscopy analysis.
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
        cnn_model = torch.load(f=model_path, map_location="cpu")

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
            min_window_size = get_min_window_size(img, config)
            config.config["piv"]["min_window_size"] = min_window_size

            # Load ROI for the current frame
            roi_i = load_frame_roi(roi=roi, frame=frame, nframes=n_frames)

            # Segment most central cell (or use ROI) to define polygon around cell. Returns (None, None) if otherwise.
            polygon, pts = get_polygon_and_roi(cell_img=cell_img, roi=roi_i, config=config)

            # Crop targets if necessary
            img, ref, cell_img, mask = create_crop_mask_targets(img, ref, cell_img, pts, crop, pad=50)

            # Perform PIV to calculate displacement vectors (u, v) for positions (x, y)
            x, y, u, v, (stack, dx, dy) = iterative_piv(img, ref, config)

            # Calculate noise value beta inside ROI, segmented cell or whole image
            beta = get_noise(config, x, y, u, v, polygon, custom_noise=custom_noise)

            # Create arrays for position (pos) and displacement vectors (vec)
            pos = np.array([x.flatten(), y.flatten()])
            vec = np.array([u.flatten(), v.flatten()])

            # Compute traction map, force field, and L_optimal
            traction_map, f_n_m, l_optimal = calculate_traction_map(
                pos,
                vec,
                beta,
                config.config["tfm"]["meshsize"],
                config.config["tfm"]["s"],
                config.config["tfm"]["pix_per_mu"],
                config.config["tfm"]["E"],
            )

            # Write results for the current frame to the HDF5 file
            results = write_frame_results(results, frame, traction_map, f_n_m, stack, cell_img, mask, beta, l_optimal,
                                          pos, vec)

        # Write metadata to the results file
        write_metadata_results(results, config.config)

        # To recover information in the future, use the following syntax:
        # h5py.File(results)['metadata'].attrs['img_path'].tobytes()

    # Return the results as a Dataset
    return Dataset(bytes_hdf5)
