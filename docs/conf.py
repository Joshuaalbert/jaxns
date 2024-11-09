# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# sys.path.insert(0, os.path.abspath(".."))  # add project root to abs path


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "jaxns"
copyright = "2024, Joshua G. Albert"
author = "Joshua G. Albert"
release = "2.6.5"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Native Sphinx extensions:
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    # Third-party extensions:
    "sphinx_rtd_theme",
    "autoapi.extension",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

suppress_warnings = [
    "autoapi.python_import_resolution"  # E.g. cyclical imports
]

add_module_names = False
modindex_common_prefix = ["jaxns."]  # So module index not all under "J"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": -1,
}

# -- Options for autodoc -----------------------------------------------------

autodoc_typehints = "description"

# -- Options for AutoAPI -----------------------------------------------------

autoapi_dirs = ["../src/jaxns"]
autoapi_root = "api"  # where to put the generated files relative to root
autoapi_options = ["members", "undoc-members", "show-inheritance",
                   "special-members", "imported-members"]
autoapi_member_order = "bysource"  # order members by source code
autoapi_ignore = ["*/tests/*"]  # ignore tests
autoapi_template_dir = "_templates/autoapi"
autoapi_python_class_content = "both"  # Use both class and __init__ docstrings
autoapi_add_toctree_entry = False

# -- Options for NBSphinx ----------------------------------------------------

nbsphinx_execute = "never"  # never execute notebooks (slow) during building

# -- Copy notebooks to docs --------------------------------------------------
# Copies the notebooks from the project directory to the docs directory so that
# they can be parsed by nbsphinx.

# Copy examples directory into docs source
# This has been replaced with a symlink
# shutil.copytree("../examples", "examples", dirs_exist_ok=True)

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}
