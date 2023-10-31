from typing import Tuple, Union, Any

import cv2
import numpy as np
import shapely
from scipy.spatial import distance
from shapely import geometry

from pytraction.net import segment as pynet
from pytraction.utils import align_slice, bead_density, normalize


# Underscore (_) is used as a placeholder for unused variable frame
def _get_reference_frame(ref_stack: np.ndarray, _, bead_channel: int) -> np.ndarray:
    """
    Extract and normalize bead channel from reference frame.
    """
    return normalize(np.array(ref_stack[bead_channel, :, :]))


def _get_img_frame(img_stack: np.ndarray, frame: int, bead_channel: int) -> np.ndarray:
    """
    Extract and normalize bead channel frame from image stack.
    """
    return normalize(np.array(img_stack[frame, bead_channel, :, :]))


def _get_cell_img(img_stack: np.ndarray, frame: int, cell_channel: int) -> np.ndarray:
    """
    Extract and normalize cell channel frame from image stack.
    """
    return normalize(np.array(img_stack[frame, cell_channel, :, :]))


def _get_raw_frames(img_stack: np.ndarray, ref_stack: np.ndarray, frame: int, bead_channel: int,
                    cell_channel: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normalized bead frame from image and reference stack get cell channels from image stack.
    """
    img = _get_img_frame(img_stack, frame, bead_channel)
    ref = _get_reference_frame(ref_stack, frame, bead_channel)
    cell_img = _get_cell_img(img_stack, frame, cell_channel)
    return img, ref, cell_img


def _get_min_window_size(img: np.ndarray, config) -> int:
    """
    Calculate window size from bead density if min_window_size is not set in pyforce config file.
    """
    if not config.config["piv"]["min_window_size"]:
        density = bead_density(img)

        # Use K-nearest neighbors (KNN) classifier to predict minimum window size based on bead density
        knn = config.knn
        min_window_size = knn.predict([[density]])
        print(f"Automatically selected window size of {min_window_size}")

        return int(min_window_size)
    else:
        return config.config["piv"]["min_window_size"]


def _load_frame_roi(roi, frame: int, nframes: int) -> list:
    """
    Load ROI for current frame.
    """
    if isinstance(roi, list):
        assert (
            len(roi) == nframes
        ), f"Warning ROI list has len {len(roi)} which is not equal to \
                                    the number of frames ({nframes}). This would suggest that you do not \
                                    have the correct number of ROIs in the zip file."
        return roi[frame]  # Return roi corresponding to active frame
    else:
        return roi  # Return single ROI (or null) which should be applied to all images


# get roi
def _cnn_segment_cell(cell_img: np.ndarray,
                      config) -> np.array:
    """
    Binarize mask for cell image.
    """
    mask = pynet.get_mask(cell_img, config.model, config.pre_fn, device=config.config["settings"]["device"],
                          )
    return np.array(mask.astype("bool"), dtype="uint8")  # Convert mask image to bool


def _detect_cell_instances_from_segmentation(mask: np.ndarray) -> Union[np.ndarray, Any]:
    """
    Detecting contours in binary mask image using simple chain approximation.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _located_most_central_cell(contours: np.ndarray,
                               mask: np.ndarray) -> np.ndarray:
    """
    Calculates the distances of all contour centres to image centre and returns the most central one.
    """
    image_center = np.asarray(mask.shape) / 2
    image_center = tuple(image_center.astype("int32"))  # Convert centre coordinates to tuple

    segmented = []
    for contour in contours:
        M = cv2.moments(contour)  # Calculate the moments of a shape up to the 3rd order

        center_X = int(M["m10"] / M["m00"])  # Divide 1st by 0th moment in the x-direction
        center_Y = int(M["m01"] / M["m00"])  # Divide 1st by 0th moment in the y-direction
        contour_center = (center_X, center_Y)

        # Calculate distance of contour centre to image centre
        distances_to_center = distance.euclidean(image_center, contour_center)

        # Save results to a list of dictionaries
        segmented.append(
            {
                "contour": contour,
                "center": contour_center,
                "distance_to_center": distances_to_center,
            }
        )

    # Sort contours by closest to farthest from image centre
    sorted_cells = sorted(segmented, key=lambda i: i["distance_to_center"])

    pts = sorted_cells[0]["contour"]  # Select the most central contour
    return pts


def _predict_roi(cell_img: np.ndarray,
                 config
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts the cell contour in an image using a CNN feed forward network and returns its coordinates.
    """
    mask = _cnn_segment_cell(cell_img, config)  # Predict binary mask for cell detection
    contours = _detect_cell_instances_from_segmentation(mask)  # Extract cell contours
    pts = _located_most_central_cell(contours, mask)  # Find most central contour
    polyx, polyy = np.squeeze(pts, axis=1).T  # Extract x and y coordinates of a contour
    return polyx, polyy


def _get_polygon_and_roi(cell_img: np.ndarray,
                         roi: list,
                         config
                         ) -> Union[type(shapely.Polygon), type(np.ndarray)]:
    """
    Segments cell using CNN model or and returns polygon shape of ROI
    """
    if config.config["settings"]["segment"]:
        polyx, polyy = _predict_roi(cell_img, config)  # Predict cell outline
        pts = np.array(list(zip(polyx, polyy)), np.int32)  # Creates array of (x,y) coordinate pairs
        polygon = geometry.Polygon(pts)  # Create polygon shape from points
        pts = pts.reshape((-1, 1, 2))  # Transforms points to a 3D 1x1x2 array
        return polygon, pts

    # If segment=True, the ROI will not be processed
    elif roi:
        polyx, polyy = roi[0], roi[1]  # Get x and y coordinates from ROI
        pts = np.array(list(zip(polyx, polyy)), np.int32)  # Creates array of (x,y) coordinate pairs
        polygon = geometry.Polygon(pts)  # Create polygon shape from points
        pts = pts.reshape((-1, 1, 2))  # Transforms points to a 3D 1x1x2 array
        return polygon, pts

    else:
        return None, None


# crop targets
def _create_mask(cell_img: np.ndarray,
                 pts: np.ndarray):
    """
    Create mask from area bordered by points on black background.
    """
    return cv2.fillPoly(np.zeros(cell_img.shape), [pts], (255))  #ToDO: Remove error message.


def _crop_roi(img: np.ndarray,
              ref: np.ndarray,
              cell_img: np.ndarray,
              pts: np.ndarray,
              pad: float = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop images to ROI defined by points and return them.
    """
    # ToDo: ErrorRemove
    cv2.polylines(cell_img, [pts], True, (255), thickness=3)  # Draws a closed polygon on the cell image

    # Calculate the coordinates and dimensions of the smallest rectangle that contains the entire polygon
    x, y, w, h = cv2.boundingRect(pts)

    # Crop original bead images to the rectangle while adding a constant padding value
    img_crop = img[y - pad:y + h + pad, x - pad:x + w + pad]
    ref_crop = ref[y - pad:y + h + pad, x - pad:x + w + pad]

    mask = _create_mask(cell_img, pts)  # Create mask from rectangle

    # Crop mask to include padding value from before
    mask_crop = mask[y - pad:y + h + pad, x - pad:x + w + pad]

    # Crop original BF image to the rectangle while adding a constant padding value
    cell_img_crop = cell_img[y - pad:y + h + pad, x - pad:x + w + pad]

    return img_crop, ref_crop, cell_img_crop, mask_crop


def _create_crop_mask_targets(img: np.ndarray,
                              ref: np.ndarray,
                              cell_img: np.ndarray,
                              pts: np.ndarray,
                              crop: bool,
                              pad: float = 50):
    """
    Create mask from shape defined by pts and crop image.
    """
    if crop and isinstance(pts, np.ndarray):
        img, ref, cell_img, mask = _crop_roi(img, ref, cell_img, pts, pad)
        return img, ref, cell_img, mask

    if not crop and isinstance(pts, np.ndarray):
        mask = _create_mask(cell_img, pts)
        return img, ref, cell_img, mask

    else:
        return img, ref, cell_img, None
