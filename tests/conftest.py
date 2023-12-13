import os


def validate_data_files():
    assert os.path.exists("data/example1/e01_pos1_axon1_ref.tif")
    assert os.path.exists("data/example1/e01_pos1_axon1.tif")

    assert os.path.exists("data/example2/1kPa-2-Position006_ref.tif")
    assert os.path.exists("data/example2/1kPa-2-Position006.tif")
    assert os.path.exists("data/example2/1kPa-2-Position006.roi")

    assert os.path.exists("data/example3/Beads3.tif")
    assert os.path.exists("data/example3/BeadsStop.tif")
    assert os.path.exists("data/example3/Cell3.tif")

    assert os.path.exists("data/matlab_data.csv")
