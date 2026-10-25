# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


project = 'MD_DL_DPP8-9'
copyright = '2026, Sam Roggeman'
author = 'Sam Roggeman'
release = '2026'
    
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme setup for alabaster
html_theme = 'alabaster'
html_theme_options = {
    'page_width': '1500px',  # Set the page width to 1200px
    'sidebar_width': '250px',  # Set the sidebar width
    'navigation_with_keys': True,  # Enable navigation with keys
}
html_css_files = ['custom.css']  # Include custom CSS file
html_static_path = ['_static']
