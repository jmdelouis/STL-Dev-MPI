# Sphinx configuration of the STL-Dev-MPI documentation: MyST Markdown and
# notebooks (myst-nb, outputs committed, never executed at build time), the
# PyData theme and the layout of GRID4EARTH/healpix-resample.
# Build with docs/build.sh (warnings are errors).
import datetime as dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_ext"))
# stl_port (the Python bindings) for autodoc; it only needs numpy to import,
# the native library is loaded when an engine is created.
sys.path.insert(0, str(HERE.parent / "python" / "stl_ensemble"))

project = "STL-Dev-MPI"
author = "STL-Dev-MPI developers"
initial_year = "2026"
year = dt.datetime.now().year
copyright = f"{initial_year}-{year}, {author}"

root_doc = "index"

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "cli_reference",
]

extlinks = {
    "issue": ("https://github.com/jmdelouis/STL-Dev-MPI/issues/%s", "GH%s"),
    "pull": ("https://github.com/jmdelouis/STL-Dev-MPI/pull/%s", "PR%s"),
}

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
myst_heading_anchors = 3

# Notebooks are run on a GPU machine and committed with their outputs.
nb_execution_mode = "off"

autosummary_generate = True
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "special-members": False, "undoc-members": False}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
# The reports and design notes are pages of the site (docs/reports.md); the
# generated pages of autosummary and of cli_reference are rebuilt every time.
exclude_patterns = ["_build", ".ipynb_checkpoints", "**/out", "jupyter_execute"]

html_theme = "pydata_sphinx_theme"
html_title = "STL-Dev-MPI"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "github_url": "https://github.com/jmdelouis/STL-Dev-MPI",
    "icon_links_label": "Quick Links",
    "navigation_depth": 3,
    "footer_start": ["copyright"],
    "footer_center": ["reports-link"],
}
