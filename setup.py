from setuptools import setup, find_packages

setup(
    name="blob_detector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib",
        "torch",
        "scikit-image",
        "pytest",
        "segmentation-models-pytorch"

    ],
)