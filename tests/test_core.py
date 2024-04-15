import os
import numpy as np
import pickle
import tempfile

from shapely.geometry import Polygon
from pytraction.core import (TractionForceConfig, process_stack)
from pytraction.tractionforcedataset import TractionForceDataset


def test_prediction_models():
    elastic_modulus = 100
    scaling_factor = 1.3
    config_path = os.path.join("example_data", "example_config.yaml")

    # No models loaded
    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=elastic_modulus,
        scaling_factor=scaling_factor,
        poisson_ratio=0.5,
        window_size=32,
        meshsize=16
    )

    assert config.cnn is None
    assert config.pre_fn is None
    assert config.knn is None

    # Models loaded
    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=elastic_modulus,
        scaling_factor=scaling_factor,
        poisson_ratio=0.5,
        window_size=0,
        meshsize=16,
        segment=True
    )

    assert config.cnn is not None
    assert config.pre_fn is not None
    assert config.knn is not None
    assert hasattr(config.knn, "predict")


def test_load_data():
    elastic_modulus = 100
    scaling_factor = 1.3
    config_path = os.path.join("example_data", "example_config.yaml")

    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=elastic_modulus,
        scaling_factor=scaling_factor,
        poisson_ratio=0.5,
        window_size=32,
        meshsize=16
    )

    # Load standard (t,c,w,h) image and (c,w,h) reference
    img_path = os.path.join("example_data", "example1_axon", "2DTFM_300Pa_RGC_Axon_TimeSeries.tif")
    ref_path = os.path.join("example_data", "example1_axon", "2DTFM_300Pa_RGC_Axon_Reference.tif")

    img, ref, roi = config.load_data(img_path, ref_path, roi_path=None, z_proj=False)

    assert img.shape == (13, 2, 362, 641)
    assert ref.shape == (2, 362, 641)

    # Load z-stack (t,z,c,w,h) image and (z,c,w,h) reference
    img_path = os.path.join("example_data", "example3_zstack", "2DTFM_300Pa_PAAGel_RGCs_TimeSeries.tif")
    ref_path = os.path.join("example_data", "example3_zstack", "2DTFM_300Pa_PAAGel_RGCs_Reference.tif")

    img, ref, roi = config.load_data(img_path, ref_path, roi_path=None, z_proj=False)

    assert img.shape == (7, 2, 512, 512)
    assert ref.shape == (2, 512, 512)

    img, ref, roi = config.load_data(img_path, ref_path, roi_path=None, z_proj=True)

    assert img.shape == (7, 2, 512, 512)
    assert ref.shape == (2, 512, 512)

    # Load ROI
    img_path = os.path.join("example_data", "example2_fibroblast", "2DTFM_10kPa_hPAAGel_3T3Fibroblasts_TimeSeries.tif")
    ref_path = os.path.join("example_data", "example2_fibroblast", "2DTFM_10kPa_hPAAGel_3T3Fibroblasts_Reference.tif")
    roi_path = os.path.join("example_data", "example2_fibroblast", "AllRoiFrames.zip")

    img, ref, roi = config.load_data(img_path, ref_path, roi_path, z_proj=False)

    assert roi is not None


def test_process_stack():
    elastic_modulus = 100
    scaling_factor = 1.3
    config_path = os.path.join("example_data", "example_config.yaml")

    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=elastic_modulus,
        scaling_factor=scaling_factor,
        poisson_ratio=0.5,
        window_size=32,
        meshsize=16
    )

    img_path = os.path.join("example_data", "example3_zstack", "2DTFM_300Pa_PAAGel_RGCs_TimeSeries.tif")
    ref_path = os.path.join("example_data", "example3_zstack", "2DTFM_300Pa_PAAGel_RGCs_Reference.tif")

    img, ref, roi = config.load_data(img_path=img_path, ref_path=ref_path, roi_path=None, z_proj=True)

    log = process_stack(img_stack=img[:, :, :, :],
                        ref_stack=ref,
                        roi=roi,
                        config=config,
                        cell_channel=0,
                        bead_channel=1,
                        crop=False,
                        noise=10)

    assert type(log) is TractionForceDataset
