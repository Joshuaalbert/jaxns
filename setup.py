#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

install_requires = [
    'jax>=0.4.25',
    'jaxlib',
    'matplotlib',
    'numpy',
    'scipy',
    'tensorflow_probability',
    'tqdm',
    'dm-haiku',
    'optax',
    'jaxopt'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='jaxns',
      version='2.5.0',
      description='Nested Sampling in JAX',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/joshuaalbert/jaxns",
      author='Joshua G. Albert',
      author_email='albert@strw.leidenuniv.nl',
      install_requires=install_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      package_dir={'': './'},
      packages=find_packages('./'),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.9',
      )
