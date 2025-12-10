# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys
import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Chatter"
current_year = datetime.datetime.now().year
copyright = f"{current_year}, Mason Youngblood"
author = "Mason Youngblood"

# Import the package so we can use its version everywhere (docs, badges, etc.)
import chatter  # noqa: E402

# Sphinx uses 'version'/'release' for substitutions like |version| and |release|
release = chatter.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinxcontrib.mermaid",
]
autosummary_generate = True

# Enable MyST features we rely on (e.g., substitutions in Markdown)
myst_enable_extensions = ["substitution"]
myst_fence_as_directive = ["mermaid"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_css_files = ["custom.css"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_theme_options = {
    "repository_url": "https://github.com/masonyoungblood/chatter",
    "use_repository_button": True,
    # Show the home page as the first entry in the left sidebar navigation
    "home_page_in_toc": True,
}
html_title = "Chatter Documentation"
html_sidebars = {"posts/*": ["sbt-sidebar-nav.html"]}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
    "member-order": "bysource",
}

# MyST substitutions so we can keep badges and text in sync with the codebase.
# Note: MyST only substitutes whole values, so we build complete HTML tags here.
python_version = (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)

# Pre-construct the full HTML image tags
python_badge = f'<img src="https://img.shields.io/badge/_python-{python_version}-440154" alt="python" />'
version_badge = f'<img src="https://img.shields.io/badge/_version-{release}-21918c" alt="version" />'
doi_badge = '<img src="https://img.shields.io/badge/_doi-TBD-fde725" alt="doi" />'

# Dynamic APA citation
apa_citation = (
    f"- Youngblood, M. ({current_year}). "
    f"Chatter: a Python library for applying information theory and AI/ML models to animal communication (v{release}). "
    f"*GitHub*. [https://github.com/masonyoungblood/chatter](https://github.com/masonyoungblood/chatter)"
)

# Dynamic BibTeX citation
bibtex_citation = f"""```bibtex
@software{{youngblood_chatter_{current_year},
   author = {{Youngblood, Mason}},
   title = {{Chatter: a Python library for applying information theory and AI/ML models to animal communication}},
   version = {{v{release}}},
   date = {{{current_year}}},
   publisher = {{GitHub}},
   url = {{https://github.com/masonyoungblood/chatter}}
}}
```"""

myst_substitutions = {
    "chatter_version": release,
    "python_version": python_version,
    "python_badge": python_badge,
    "version_badge": version_badge,
    "doi_badge": doi_badge,
    "apa_citation": apa_citation,
    "bibtex_citation": bibtex_citation,
}
