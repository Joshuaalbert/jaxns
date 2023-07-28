#!/bin/bash

# remove any previously created distribution files
rm -rf dist/

# create a new virtual environment
python -m venv env
source env/bin/activate

# upgrade pip, setuptools, wheel, and twine
pip install --upgrade pip setuptools wheel twine

# build the project
python setup.py sdist bdist_wheel

# check and upload
twine check dist/* && twine upload dist/*

# deactivate and remove the virtual environment
deactivate
rm -rf env/
