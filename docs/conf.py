import os
import sys

project = "bihalofit-jax"
copyright = "2026, Alexander Roskill"
author = "Alexander Roskill"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
]

# baccoemu needs GSL / a real install to import; mock it so the docs build
# doesn't require the full scientific stack.
autodoc_mock_imports = ["baccoemu"]

autosummary_generate = True
numpydoc_show_class_members = False

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*notes*.md", "README.md"]

html_theme = "pydata_sphinx_theme"
html_title = "bihalofit-jax"
html_theme_options = {
    "github_url": "https://github.com/AlexRoskill/bihalofit-jax",
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
        "text": "bihalofit-jax",
    },
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
