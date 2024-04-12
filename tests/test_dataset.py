import os
import pandas as pd

from pytraction.tractionforcedataset import TractionForceDataset


def test_read_tfm_results():
    dataset_path = os.path.join("example_data", "example5_tractiondataset", "example_tractiondataset.h5")
    traction_dataset = TractionForceDataset(dataset_path)

    assert len(traction_dataset) == 1

    assert list(traction_dataset.columns) == [
        'cell_image',
        'deformation',
        'deformation_interpolated',
        'drift_corrected_stack',
        'frame',
        'mask_roi',
        'noise_beta',
        'optimal_lambda',
        'position',
        'position_interpolated',
        'traction'
    ]

    assert isinstance(traction_dataset[0], pd.core.frame.DataFrame)
    for col in traction_dataset.columns:
        assert isinstance(traction_dataset[col], pd.core.frame.DataFrame)
