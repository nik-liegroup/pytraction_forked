import os
import zipfile
from typing import Tuple, Union

import pandas as pd
from read_roi import read_roi_file


def _recursive_lookup(key: str, dictionary: dict) -> Union[list, None]:
    """
    Given a nested dictionary d, return the first instance of d[k].

    @param  key: Target key
    @param  dictionary: Nested dictionary
    """
    if key in dictionary:
        return dictionary[key]
    for v in dictionary.values():
        if isinstance(v, dict):
            a = _recursive_lookup(key, v)
            if a is not None:
                return a
    return None


def _load_csv_roi(roi_path: str) -> Tuple[list, list]:
    """
    Load ROI data from .csv file containing two column specifying x and y values.

    @param  roi_path: System path to .csv file containing ROI information
    """
    x, y = pd.read_csv(roi_path).T.values
    return list(x), list(y)


def _load_roireader_roi(roi_path: str) -> Tuple[list, list]:
    """
    Load ROI data from .roi file.

    @param  roi_path: System path to .roi file containing ROI information
    """
    # ToDo: Check what this function does
    d = read_roi_file(roi_path)

    # Return x and y coordinates from dictionary
    x = _recursive_lookup("x", d)
    y = _recursive_lookup("y", d)

    return x, y


def _load_zip_roi(roi_path: str) -> list:
    """
    Load ROI data from .zip folder.

    @param  roi_path: System path to .zip folder containing ROI information
    """
    rois = []
    # Open .zip folder and extract each sub-file
    with zipfile.ZipFile(roi_path) as zip_roi:
        for file in zip_roi.namelist():
            roi_path_file = zip_roi.extract(file)

            # ToDo: Check what this function does
            d = read_roi_file(roi_path_file)

            # Return x and y coordinates from dictionary
            x = _recursive_lookup("x", d)
            y = _recursive_lookup("y", d)

            # Append x and y coordinates to array
            rois.append((x, y))

            # Delete extracted file
            os.remove(roi_path_file)
    return rois


def roi_loaders(roi_path) -> Union[Tuple[list, list], list, None]:
    """
    Call appropriate import functions depending on roi file format.

    @param  roi_path: System path to file containing ROI information (.csv, .roi or .zip)
    """
    if ".csv" in roi_path:
        return _load_csv_roi(roi_path)

    elif ".roi" in roi_path:
        return _load_roireader_roi(roi_path)

    elif ".zip" in roi_path:
        return _load_zip_roi(roi_path)

    elif "" == roi_path:
        return None

    else:
        msg = "The roi loader expects files in '.csv', '.roi', or '.zip' format"
        raise NotImplementedError(msg)
