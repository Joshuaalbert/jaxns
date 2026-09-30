#!/bin/bash

# remove any previously created distribution files
rm -rf dist/

# create a new virtual environment
python -m venv env
source env/bin/activate

# install the standards-based build and upload tools
pip install --upgrade pip build twine

# build the project
python -m build

# check and upload
twine check dist/* && twine upload dist/*

# deactivate and remove the virtual environment
deactivate
rm -rf env/
