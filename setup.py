from setuptools import setup

VERSION = "0.1.1"


DESCRIPTION = "Bayesian Traction Force Microscopy"


CLASSIFIERS = [
    "Natural Language :: English",
    "Programming Language :: Python :: 3.8",
]

REQUIREMENTS = [
    "albumentations<=1.4.3",
    "attrs==23.2.0",
    "certifi==2024.2.2",
    "cycler==0.12.1",
    "Cython==3.0.10",
    "decorator==5.1.1",
    "gitdb==4.0.11",
    "GitPython==3.1.43",
    "h5py==3.11.0",
    "imageio==2.34.0",
    "iniconfig==2.0.0",
    "kiwisolver==1.4.5",
    "matplotlib==3.7.5",
    "natsort==8.4.0",
    "networkx==3.1",
    "numpy==1.24.4",
    "opencv-python==4.9.0.80",
    "OpenPIV==0.25.2",
    "packaging==24.0",
    "pandas==2.0.3",
    "Pillow==10.3.0",
    "pluggy==1.4.0",
    "py==1.11.0",
    "pyparsing==3.1.2",
    "pytest==8.1.1",
    "python-dateutil==2.9.0",
    "pytz==2024.1",
    "PyWavelets==1.4.1",
    "PyYAML==6.0.1",
    "read-roi==1.6.0",
    "requests==2.31.0",
    "scikit-image<=0.21.0",
    "scikit-learn==0.24.1",
    "scipy==1.10.1",
    "segmentation_models_pytorch==0.3.3",
    "shapely==2.0.3",
    "smmap==5.0.1",
    "tifffile==2023.7.10",
    "toml==0.10.2",
    "tqdm==4.66.2",
]


SETUP_REQUIRES = ("pytest-cov", "pytest-runner", "pytest", "codecov")
TESTS_REQUIRES = ("pytest-cov", "codecov")


PACKAGES = [
    "pytraction",
    "pytraction.net",
]


options = {
    "name": "pytraction",
    "version": VERSION,
    "author": "Ryan Greenhalgh & Niklas Gampl",
    "author_email": "rdg31@cam.ac.uk, niklas.gampl@fau.de",
    "description": DESCRIPTION,
    "classifiers": CLASSIFIERS,
    "packages": PACKAGES,
    "include_package_data": True,
    "package_data": {'pytraction': ['*.pth', '*.pickle',]},
    "setup_requires": SETUP_REQUIRES,
    "test_requires": TESTS_REQUIRES,
    "install_requires": REQUIREMENTS,
    "entry_points": {
        "console_scripts": ["pytraction_get_data=pytraction.get_example_data:main"]
    }
}
setup(**options)
