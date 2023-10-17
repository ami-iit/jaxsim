# Configuration file for the Sphinx documentation builder.
import os
import pathlib
import sys

# -- Version information

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../"))

module_path = os.path.abspath("../src/")
sys.path.insert(0, module_path)
version_file = os.path.abspath("../src/jaxsim/version.txt")

with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

# -- Project information

project = "JAXsim"
copyright = "2022, Artificial and Mechanical Intelligence"
author = "Artificial and Mechanical Intelligence"

release = f"{__version__}"
version = f"main ({__version__})"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_multiversion",
    # "sphinx_fontawesome",
    # "breathe",
    # "sphinx_tabs.tabs",
]

# -- Options for intersphinx extension

language = "en"

templates_path = ["_templates"]

master_doc = "index"

exclude_patterns = ["_build"]

autodoc_typehints = "signature"

# autodoc_default_options = {
#     "members": True,
#     "undoc-members": True,
#     "member-order": "bysource",
# }

# # Napoleon settings
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = True
# napoleon_use_param = True
# napoleon_use_rtype = True

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output

epub_show_urls = "footnote"
