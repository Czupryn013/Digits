#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name="digits",
      version="0.1.0",
      packages=find_packages(),
      install_requires = ["flask", "numpy", "pygame", "matplotlib", "tensorflow", "scikit-learn"]
      )