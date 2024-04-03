# Configuration file for the Sphinx documentation builder.
import os
import sys

from pkg_resources import get_distribution

if os.environ.get("READTHEDOCS"):
    checkout_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    os.environ["CONDA_PREFIX"] = os.path.realpath(
        os.path.join("..", "..", "conda", checkout_name)
    )

import jaxsim

# -- Version information

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../"))

module_path = os.path.abspath("../src/")
sys.path.insert(0, module_path)

__version__ = get_distribution("jaxsim").version

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
    "enum_tools.autoenum",
    "sphinx_design",
]

# -- Options for intersphinx extension

language = "en"

html_theme = "sphinx_book_theme"

templates_path = ["_templates"]

html_title = f"JAXsim {version}"

master_doc = "index"

autodoc_typehints_format = "short"

exclude_patterns = ["_build"]

autodoc_typehints = "signature"

autosummary_generate = False

epub_show_urls = "footnote"
