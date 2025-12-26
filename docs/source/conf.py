# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

import pydata_sphinx_theme
from sphinx.application import Sphinx
from sphinx.locale import _

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PKG_DIR = SRC_DIR / "topo_metrics"

sys.path.insert(0, str(SRC_DIR))

ANNOUNCEMENT = Path(__file__).parent.joinpath(
    "_templates", "announcement.html"
).read_text(encoding="utf-8")


# -- Project information -----------------------------------------------------

project = 'topo-metrics'
copyright = '2025, Thomas C. Nicholas'
author = 'Thomas C. Nicholas'

version = '0.1.4'
release = '0.1.4'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinx_design",
    "sphinx_copybutton",
    "autoapi.extension",
    "jupyter_sphinx",
    "nbsphinx",
    "sphinx_togglebutton",
    "sphinx_favicon",
    "pydata_sphinx_theme",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
intersphinx_mapping = {"sphinx": ("https://www.sphinx-doc.org/en/master", None)}

# -- Sitemap -----------------------------------------------------------------

if not os.environ.get("READTHEDOCS"):
    extensions += ["sphinx_sitemap"]
    html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
    sitemap_locales = [None]
    sitemap_url_scheme = "{link}"

# -- MyST options ------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "linkify", "substitution"]
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# -- Internationalisation ----------------------------------------------------

language = "en"

# -- sphinx_togglebutton options ---------------------------------------------
togglebutton_hint = str(_("Click to expand"))
togglebutton_hint_hide = str(_("Click to collapse"))

# -- Sphinx-copybutton options ---------------------------------------------
copybutton_exclude = ".linenos, .gp"
copybutton_selector = ":not(.prompt) > div.highlight pre"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "TopoMetrics"
html_sourcelink_suffix = ""
html_logo = ""
html_favicon = ""
html_last_updated_fmt = ""

html_theme_options = {
    "external_links": [],
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/tcnicholas/topo-metrics",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/topo-metrics",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome",
        },
        {
            "name": "Bluesky",
            "url": "https://bsky.app/profile/tcnicholas.bsky.social",
            "icon": "fa-brands fa-bluesky",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "navbar_align": "left",
    #"announcement": ANNOUNCEMENT,
    "show_version_warning_banner": True,
    "navbar_center": ["navbar-nav"],
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
    "secondary_sidebar_items": {
        "**/*": ["page-toc", "edit-this-page", "sourcelink"],
        "examples/no-sidebar": [],
    },
    "back_to_top_button": True,
}

html_context = {
    "github_user": "tcnicholas",
    "github_repo": "topo-metrics",
    "github_version": "main", 
    "doc_path": "docs",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [
    ("custom-icons.js", {"defer": "defer"}),
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

# -- AutoAPI -----------------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = [str(PKG_DIR)]
autoapi_root = "_autoapi"
autoapi_add_toctree_entry = True
autoapi_keep_files = True

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "show-module-annotations", 
]
autosummary_generate = False
autoapi_python_use_annotated = True
autoapi_python_class_content = "both"