"""Command-line reference generated from the usage strings in the Rust code.

Every binary prints its options from one `USAGE` string (`--help`); this
module reads those strings from the sources and writes one page per binary
into `docs/api/cli/` when Sphinx starts, so the reference cannot drift from
the code.  Binaries without a `USAGE` string (`stl-minkowski`,
`stl-mpi-check`) are documented from the `//!` header of their source.

    python docs/_ext/cli_reference.py          # writes the pages without Sphinx
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "api" / "cli"

# (binary, source, constant or None for the //! header, summary, feature notes)
BINARIES = [
    ("stl-synth", "crates/stl-cli/src/lib.rs", "USAGE",
     "Synthesis of one HEALPix map, or of an ensemble, from the statistics of a target map.",
     "Guide: {doc}`../../user-guide/stl-synth`."),
    ("stl-stats", "crates/stl-cli/src/lib.rs", "STATS_USAGE",
     "Scattering-covariance statistics of HEALPix maps (JSON: plan and values).",
     "Guide: {doc}`../../user-guide/stl-stats`. The engine options have the meaning they have in {doc}`stl-synth`."),
    ("stl-ensemble", "crates/stl-cli/src/ensemble/cli.rs", "USAGE",
     "Ensemble synthesis with P. Campeti's method: calibration on reference maps, then synthesis.",
     "Guide: {doc}`../../user-guide/stl-ensemble`. The GPU options have the meaning they have in {doc}`stl-synth`."),
    ("stl-synth2d", "crates/stl-cli/src/bin/stl-synth2d.rs", "USAGE",
     "Synthesis of periodic planar (2D) images.",
     "Guide: {doc}`../../user-guide/stl-synth2d`."),
    ("stl-minkowski", "crates/stl-cli/src/bin/stl-minkowski.rs", None,
     "Minkowski functionals per pyramid level, with the estimator of `stl-synth --minkowski-weight`.",
     ""),
    ("stl-mpi-check", "crates/stl-mpi/src/bin/stl-mpi-check.rs", None,
     "Distributed statistics and gradients of an STL-Dev fixture under MPI, compared with the fixture.",
     "Built with the `mpi` feature of `stl-mpi`. Guide: {doc}`../../user-guide/mpi`."),
]


def usage_string(text, name):
    """The body of `const NAME: &str = "\\ ... ";` with Rust's line continuation removed."""
    m = re.search(r'const ' + name + r': &str = "\\\n(.*?)";', text, re.S)
    if not m:
        raise ValueError(f"no {name} string")
    return m.group(1).replace('\\"', '"')


def header_doc(text):
    """The `//!` header of a source file, without the markers."""
    lines = []
    for line in text.splitlines():
        if not line.startswith("//!"):
            break
        lines.append(line[4:] if line.startswith("//! ") else line[3:])
    return "\n".join(lines).strip()


def description_column(usage):
    """The column where the descriptions start: the most common start after an
    option followed by two spaces or more."""
    cols = {}
    for line in usage.splitlines():
        m = re.match(r"^(\s{1,4}--\S.*?\s{2,})\S", line)
        if m:
            cols[len(m.group(1))] = cols.get(len(m.group(1)), 0) + 1
    return max(cols, key=cols.get) if cols else 24


def options(usage):
    """(section, [(option, description)]) from a usage text: an option line starts
    with `-` indented by at most 4 spaces; deeper lines continue its description;
    `name:` lines start a section.  The option ends at the description column
    when two spaces precede it there, or one space and a description of several
    words follows; else at the first run of two spaces."""
    col = description_column(usage)
    sections, current, entries = [], "", []
    for line in usage.splitlines():
        if re.match(r"^[a-z][a-z ]*:\s*$", line):
            if entries:
                sections.append((current, entries))
            current, entries = line.strip().rstrip(":"), []
            continue
        if re.match(r"^\s{1,4}-", line):
            rest = line[col:]
            if len(line) > col and line[col - 1] == " " and rest[0] != " " and (line[col - 2] == " " or " " in rest.strip()):
                opt, desc = line[:col], line[col:]
            else:
                parts = re.split(r"\s{2,}", line.strip(), maxsplit=1)
                opt, desc = parts[0], parts[1] if len(parts) > 1 else ""
            entries.append([opt.strip(), desc.strip()])
        elif entries and line.startswith(" " * 6) and line.strip():
            entries[-1][1] = (entries[-1][1] + " " + line.strip()).strip()
    if entries:
        sections.append((current, entries))
    return sections


def cell(text):
    return text.replace("|", "\\|")


def page(binary, source, const, summary, notes):
    text = (ROOT / source).read_text(encoding="utf-8")
    out = [f"# `{binary}`", "", summary, ""]
    if notes:
        out += [notes, ""]
    if const:
        usage = usage_string(text, const)
        out += [f"Generated from `{const}` in `{source}` (the text of `{binary} --help`).", ""]
        for section, entries in options(usage):
            out += [f"## {section.capitalize() if section else 'Options'}", "", "| Option | Description |", "|---|---|"]
            out += [f"| `{cell(o)}` | {cell(d)} |" for o, d in entries]
            out.append("")
        out += ["## Full text", "", "```text", usage.rstrip(), "```", ""]
    else:
        out += [f"Generated from the header of `{source}`.", "", header_doc(text), ""]
    return "\n".join(out)


def write_pages(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    for b in BINARIES:
        (out / f"{b[0]}.md").write_text(page(*b), encoding="utf-8")
    index = ["# Command-line reference", "",
             "Every option of every binary, generated at build time from the usage strings in",
             "the code (`docs/_ext/cli_reference.py`).", "",
             "| Binary | Does |", "|---|---|"]
    index += [f"| {{doc}}`{b[0]}` | {b[3]} |" for b in BINARIES]
    index += ["", "```{toctree}", ":hidden:", ""] + [b[0] for b in BINARIES] + ["```", ""]
    (out / "index.md").write_text("\n".join(index), encoding="utf-8")


def setup(app):
    app.connect("builder-inited", lambda app: write_pages())
    return {"version": "1", "parallel_read_safe": True}


if __name__ == "__main__":
    write_pages()
    print("wrote", OUT)
