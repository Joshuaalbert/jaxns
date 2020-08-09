#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_numpy_version__ = '1.16.2'
__minimum_jax_version__ = '0.1.67'

setup_requires = ['numpy>=' + __minimum_numpy_version__,
                  'jax>=' + __minimum_jax_version__]

setup(name='jaxns',
      version='0.0.1',
      description='Nested Sampling in JAX',
      author=['Joshua G. Albert'],
      author_email=['albert@strw.leidenuniv.nl'],
      setup_requires=setup_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      # package_data= {'born_rime':['arrays/*', 'data/*']},
      package_dir={'': './'},
      packages=find_packages('./')
      )
