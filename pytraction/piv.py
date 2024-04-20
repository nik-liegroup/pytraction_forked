import numpy as np
import cv2
from typing import Tuple
from openpiv import pyprocess, validation, filters, scaling, tools
from openpiv.windef import *
from pytraction.widim import WiDIM
from pytraction.regularization import *


def extended_area_piv(img: np.ndarray,
                      ref: np.ndarray,
                      config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute deformation field using particle image velocimetry (PIV) with an implementation of extended search area.
    """
    config = config.config

    # Calculate window overlap in pixels
    overlap = int(config["piv"]["overlap_ratio"] * config["piv"]["window_size"])

    # Compute displacement field using a standard PIV cross-correlation algorithm
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        frame_a=ref.astype(np.int32),
        frame_b=img.astype(np.int32),
        window_size=config["piv"]["window_size"],
        overlap=overlap,
        dt=config["piv"]["dt"],
        search_area_size=config["piv"]["window_size"],
        correlation_method=config["piv"]["correlation_method"],
        subpixel_method=config["piv"]["subpixel_method"],
        sig2noise_method=config["piv"]["sig2noise_method"],
        width=config["piv"]["width"],
        normalized_correlation=config["piv"]["normalized_correlation"],
        use_vectorized=config["piv"]["use_vectorized"]
    )

    # Return coordinates for PIV vector field
    x0, y0 = pyprocess.get_coordinates(image_size=ref.shape,
                                       search_area_size=config["piv"]["window_size"],
                                       overlap=overlap)

    # Remove vectors in field which have a signal-to-noise ratio larger than the threshold
    flags = validation.sig2noise_val(sig2noise, threshold=config["piv"]["sig2noise_threshold"])
    u_f, v_f = filters.replace_outliers(u0, v0, flags,
                                        method=config["piv"]["outliers_method"],
                                        max_iter=3,
                                        kernel_size=2)

    # Transform from image to physical coordinates
    x, y, u, v = tools.transform_coordinates(x0, y0, u_f, v_f)

    return x, y, u, v


def widim_piv(img: np.ndarray,
              ref: np.ndarray,
              config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute deformation field using particle image velocimetry (PIV) with an implementation of window size coarsening.
    """
    config = config.config

    try:
        # Compute displacement field using a window deformation PIV algorithm
        x, y, u, v, mask = WiDIM(
            frame_a=ref.astype(np.int32),
            frame_b=img.astype(np.int32),
            mark=np.ones_like(ref).astype(np.int32),
            min_window_size=config["piv"]["window_size"],
            overlap_ratio=config["piv"]["overlap_ratio"],
            coarse_factor=config["piv"]["coarse_factor"],
            dt=config["piv"]["dt"],
            validation_method=config["piv"]["validation_method"],
            trust_1st_iter=config["piv"]["trust_1st_iter"],
            validation_iter=config["piv"]["validation_iter"],
            tolerance = config["piv"]["tolerance"],
            nb_iter_max = config["piv"]["nb_iter_max"],
            subpixel_method = config["piv"]["subpixel_method"],
            sig2noise_method = config["piv"]["sig2noise_method"],
            sig2noise_threshold = config["piv"]["sig2noise_threshold"],
            width = config["piv"]["width"]
        )
        return x, y, u, v

    except Exception as e:
        if isinstance(e, ZeroDivisionError):
            # Reduce window size and call compute_piv again
            config_tmp = config.copy()
            config_tmp["piv"]["window_size"] = (
                    config["piv"]["window_size"] // 2
            )
            print(f"Reduced min window size to {config_tmp['piv']['window_size'] // 2} in recursive call.")
            return widim_piv(img, ref, config_tmp)
        else:
            raise e


def windef_piv(frame_a, frame_b, image_mask, counter, config):
    """
    Compute deformation field using particle image velocimetry (PIV) with an implementation of window size coarsening.
    """
    settings = windef.PIVSettings()

    # Calculate first pass
    x, y, u, v, s2n = first_pass(
        frame_a,
        frame_b,
        settings
    )

    if image_mask is None:
        grid_mask = np.zeros_like(u, dtype=bool)
    else:
        grid_mask = scn.map_coordinates(image_mask, [y, x]).astype(bool)

    # Mask the velocity
    u = np.ma.masked_array(u, mask=grid_mask)
    v = np.ma.masked_array(v, mask=grid_mask)

    if settings.validation_first_pass:
        flags = validation.typical_validation(u, v, s2n, settings)
    else:
        flags = np.zeros_like(u, dtype=bool)

    # Filter to replace the values that where marked by the validation
    if (settings.num_iterations == 1 and settings.replace_vectors) or (settings.num_iterations > 1):
        # for multi-pass we cannot have holes in the data
        # after the first pass
        u, v = filters.replace_outliers(
            u,
            v,
            flags,
            method=settings.filter_method,
            max_iter=settings.max_filter_iteration,
            kernel_size=settings.filter_kernel_size,
        )

    # Adding masks to add the effect of all the validations
    if settings.smoothn:
        u, *_ = smoothn.smoothn(
            u,
            s=settings.smoothn_p
        )
        v, *_ = smoothn.smoothn(
            v,
            s=settings.smoothn_p
        )

        # enforce grid_mask that possibly destroyed by smoothing
        u = np.ma.masked_array(u, mask=grid_mask)
        v = np.ma.masked_array(v, mask=grid_mask)

    # Multi pass
    for i in range(1, settings.num_iterations):
        x, y, u, v, grid_mask, flags = multipass_img_deform(
            frame_a,
            frame_b,
            i,
            x,
            y,
            u,
            v,
            settings,
        )

        # If the smoothing is active, we do it at each pass but not the last one
        if settings.smoothn is True and i < settings.num_iterations - 1:
            u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(
                u, s=settings.smoothn_p
            )
            v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(
                v, s=settings.smoothn_p
            )
        if not isinstance(u, np.ma.MaskedArray):
            raise ValueError('Not a masked array anymore')

        if image_mask is not None:
            grid_mask = scn.map_coordinates(image_mask, [y, x]).astype(bool)
            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)
        else:
            u = np.ma.masked_array(u, np.ma.nomask)
            v = np.ma.masked_array(v, np.ma.nomask)

    # We now use only 0s instead of the image masked regions
    u = u.filled(0.)
    v = v.filled(0.)

    if image_mask is not None:
        grid_mask = scn.map_coordinates(image_mask, [y, x]).astype(bool)
        u = np.ma.masked_array(u, mask=grid_mask)
        v = np.ma.masked_array(v, mask=grid_mask)
    else:
        u = np.ma.masked_array(u, np.ma.nomask)
        v = np.ma.masked_array(v, np.ma.nomask)

    # pixel/frame -> pixel/second
    u /= settings.dt
    v /= settings.dt

    # Scale pixels to microns
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=settings.scaling_factor)

    x, y, u, v = transform_coordinates(x, y, u, v)

    return x, y, u, v


"""
# All the PIV settings for the batch analysis with multi-processing and window deformation.

filepath_images: Union[pathlib.Path, str]
save_path: pathlib.Path
save_folder_suffix: str
frame_pattern_a: str
frame_pattern_b: str
roi: Union[Tuple[int, int, int, int], str] # Region of interest: (xmin, xmax, ymin, ymax) or 'full'

dynamic_masking_method: Optional[str] # 'edge' or 'intensity'
dynamic_masking_threshold: float
dynamic_masking_filter_size: int
static_mask: Optional[np.ndarray] # Boolean matrix of image shape

correlation_method: str # 'circular' or 'linear'
normalized_correlation: bool
windowsizes: Tuple[int, ...] # Interroagtion window size for each pass.
overlap: Tuple[int, ...]  # Interroagtion window size overlap for each pass.
num_iterations: int  # Should be equal len(windowsizes)

subpixel_method: str # 'gaussian', 'centroid' or 'parabolic'
use_vectorized: bool
deformation_method: str # 'symmetric' or 'second image'
interpolation_order: int
scaling_factor: float
dt: float

sig2noise_method: Optional[str] # 'peak2mean', 'peak2peak' or 'None'
sig2noise_mask: int
sig2noise_threshold: float
sig2noise_validate: bool
validation_first_pass: bool

min_max_u_disp: Tuple
min_max_v_disp: Tuple
std_threshold: int
median_threshold: int
median_size: int

replace_vectors: bool
smoothn: bool
smoothn_p: float
filter_method: str # 'localmean', 'disk' or 'distance'
max_filter_iteration: int
filter_kernel_size: int

save_plot: bool
show_plot: bool
scale_plot: int
show_all_plots: bool
invert: bool
fmt: str
"""
