#!/bin/bash

pip install wheel twine

python setup.py sdist bdist_wheel

twine check dist/* && twine upload dist/*