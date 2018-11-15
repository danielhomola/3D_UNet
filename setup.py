from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project implements the 3D U-Net paper using Tensorflow and trains it on prostate cancer data from the 2013 NCI-ISBI Challenge.',
    author='Daniel Homola',
    license='BSD-3',
)
