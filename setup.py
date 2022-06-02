#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_jax_version__ = '0.2.9'

setup_requires = ['jax>=' + __minimum_jax_version__]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='jaxns',
      version='1.1.0',
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
