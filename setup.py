#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__maximum_numpy_version__ = '1.19.0'
__minimum_jax_version__ = '0.1.67'

setup_requires = ['numpy<' + __maximum_numpy_version__,
                  'jax>=' + __minimum_jax_version__]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='jaxns',
      version='0.0.1',
      description='Nested Sampling in JAX',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/joshuaalbert/jaxns",
      author='Joshua G. Albert',
      author_email='albert@strw.leidenuniv.nl',
      setup_requires=setup_requires,
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
      python_requires='>=3.6',
      )
