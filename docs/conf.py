# Configuration file for the Sphinx documentation builder.
import pathlib
import sys

# -- Version information

sys.path.insert(0, pathlib.Path(__file__).parents[2].absolute())
version_file = (
    pathlib.Path(__file__).parents[1].absolute() / "src" / "jaxsim" / "version.txt"
)

with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

# -- Project information

project = "JAXsim"
copyright = "2022, Artificial and Mechanical Intelligence"
author = "Artificial and Mechanical Intelligence"

release = "__version__"
version = "main (" + __version__ + ")"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
]

# -- Options for intersphinx extension

language = "en"

templates_path = ["_templates"]

master_doc = "index"

exclude_patterns = ["_build"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output

epub_show_urls = "footnote"
