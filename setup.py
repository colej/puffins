from setuptools import setup, find_packages, Extension
import numpy as np


# Automatically include all Python packages
packages = find_packages()

setup(
    name='puffins',
    version='0.1.0',
    description='Tools for solving some problems that are completely unrelated to puffins, but are still interesting for poeple who like multiply periodic time series data.',
    author='Cole Johnston, Nora Eisner, David W. Hogg',
    author_email='colej@mpa-garching.mpg.de',
    url='https://github.com/colej/puffins',
    packages=packages,
    # package_dir={"puffins": "puffins",},
    # packages=["puffins"],
    install_requires=[
        # List your project's dependencies here
    ],
    classifiers=[
        # Choose appropriate classifiers from:
        # https://pypi.org/classifiers/
    ],
)