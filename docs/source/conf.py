"""Configuration file for the Sphinx documentation builder."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

from deeplc import __version__

# Project information
project = "deeplc"
author = "CompOmics"
github_project_url = "https://github.com/compomics/deeplc/"
github_doc_root = "https://github.com/compomics/deeplc/tree/main/docs/"

# Version
release = __version__

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_mdinclude",
    "sphinx_click",
]
source_suffix = [".rst", ".md"]
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# Options for HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/img/program_icon.png"

# Autodoc options
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "init"

# Intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "psm_utils": ("https://psm-utils.readthedocs.io/en/stable/", None),
}


def setup(app):
    config = {  # noqa: F841
        # "auto_toc_tree_section": "Contents",
        "enable_eval_rst": True,
    }
