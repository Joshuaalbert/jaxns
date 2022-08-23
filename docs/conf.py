# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, glob, shutil, sphinx_rtd_theme

# sys.path.insert(0, os.path.abspath(".."))  # add project root to abs path


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'jaxns'
copyright = '2022, Joshua Albert'
author = 'Joshua Albert'
release = '1.1.1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

suppress_warnings = [
    "autoapi.python_import_resolution"  # E.g. cyclical imports
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_theme_options = {
    "navigation_depth": -1,
}

# -- Options for AutoAPI -----------------------------------------------------

autoapi_dirs = ['../jaxns']
autoapi_root = 'api'  # where to put the generated files relative to root
autoapi_options =  ['members', 'undoc-members', 'show-inheritance', 
                    'special-members', 'imported-members']
autoapi_member_order = 'bysource'  # order members by source code
autoapi_ignore = ["*/tests/*"]  # ignore tests
autoapi_template_dir = '_templates/autoapi'
autoapi_python_class_content = "both"  # Use both class and __init__ docstrings
autoapi_add_toctree_entry = False


# -- Options for NBSphinx ----------------------------------------------------

nbsphinx_execute = "never"  # never execute notebooks (slow) during building


# -- Copy notebooks to docs --------------------------------------------------
# Copies the notebooks from the project directory to the docs directory so that
# they can be parsed by nbsphinx.

if not os.path.exists("examples"):
    os.makedirs("examples")
    
for file in glob.glob("../examples/*/*.ipynb"):
    # os.system("cp {} examples".format(file))
    # os.path.join("examples", os.path.basename(file))
    shutil.copy(file, "examples/")
