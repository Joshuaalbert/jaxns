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


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_theme_options = {
    # "collapse_navigation": False,
}

# -- Options for AutoAPI -----------------------------------------------------

autoapi_dirs = ['../jaxns']
autoapi_root = 'api'
autoapi_options =  [ 'members', 'undoc-members', 'show-inheritance', 'special-members', 'imported-members',]
autoapi_member_order = 'bysource'
autoapi_ignore = ["*/tests/*"]
autoapi_template_dir = '_templates/autoapi'


# -- Options for NBSphinx ----------------------------------------------------

nbsphinx_execute = "never"


# -- Copy notebooks to docs --------------------------------------------------

if not os.path.exists("examples"):
    os.makedirs("examples")
    
for file in glob.glob("../examples/*/*.ipynb"):
    # os.system("cp {} examples".format(file))
    # os.path.join("examples", os.path.basename(file))
    shutil.copy(file, "examples/")
