# Sphinx configuration of the STL-Dev-MPI documentation (MyST Markdown,
# the layout of GRID4EARTH/healpix-resample).  Build with docs/build.sh.
import datetime as dt

project = "STL-Dev-MPI"
author = "STL-Dev-MPI developers"
initial_year = "2026"
year = dt.datetime.now().year
copyright = f"{initial_year}-{year}, {author}"

root_doc = "index"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.extlinks",
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

templates_path = ["_templates"]
# The reports (USAGE.md, gpu.md, ...) are pages of the site; the notes that
# are not linked from it stay in the repository only.
exclude_patterns = ["_build", ".ipynb_checkpoints", "figures/README.md"]
suppress_warnings = ["myst.xref_missing", "myst.header"]

html_theme = "pydata_sphinx_theme"
html_title = "STL-Dev-MPI"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "github_url": "https://github.com/jmdelouis/STL-Dev-MPI",
    "icon_links_label": "Quick Links",
    "navigation_depth": 3,
}
