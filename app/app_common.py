"""Helpers shared by the app pages: where the saved files are, how to read them, and a few display helpers.

Everything read here is an aggregate file that ships with the repository (reports/, reports/figures/, models/).
Nothing reads row-level data, nothing is computed from the dataset, and nothing in the app writes a file.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FIGURES = "reports/figures"

# The approved disclaimer, word for word. It is shown at the top of every page.
DISCLAIMER = (
    "This application is for educational purposes only. It is not a medical device and does not provide a "
    "diagnosis or medical advice. Its predictions should not be used to make healthcare decisions. Please consult "
    "a qualified healthcare professional if you have concerns about diabetes."
)

RESULTS_REPORT = "reports/09_model_results.md"
EDA_REPORT = "reports/05_eda_training.md"


class ReportError(Exception):
    """A saved report or figure is missing or does not look as expected. The message is safe to show."""


@contextmanager
def guard(what: str):
    """Show a warning instead of crashing the page when a saved file is missing or malformed."""
    try:
        yield
    except ReportError as exc:
        st.warning(f"{what} is unavailable: {exc}.", icon=":material/error:")


# --- reading saved files ---------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_text(relative: str) -> str:
    try:
        return (ROOT / relative).read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"{relative} could not be read") from exc


@st.cache_data(show_spinner=False)
def read_json(relative: str) -> dict:
    try:
        return json.loads(read_text(relative))
    except json.JSONDecodeError as exc:
        raise ReportError(f"{relative} is not valid JSON") from exc


# --- tables and bullet lists inside the Markdown reports ------------------------------------------------
# The app shows the report's own tables, so what a visitor sees is exactly what the report says.
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _cells(line: str) -> list[str]:
    """The cells of one Markdown table row, without the bold markers the report puts around the chosen model."""
    return [cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")]


@st.cache_data(show_spinner=False)
def markdown_tables(relative: str) -> list[dict]:
    """Every pipe table in a Markdown file, with the headings above it."""
    tables: list[dict] = []
    headings: dict[int, str] = {}
    block: list[str] = []

    def close() -> None:
        nonlocal block
        if len(block) >= 3:  # header, separator line, at least one row
            header = _cells(block[0])
            rows = [_cells(line) for line in block[2:]]
            if any(len(row) != len(header) for row in rows):
                raise ReportError(f"{relative} has a table with ragged rows")
            tables.append({"headings": [headings[level] for level in sorted(headings)],
                           "frame": pd.DataFrame(rows, columns=header)})
        block = []

    for line in read_text(relative).splitlines():
        text = line.strip()
        if text.startswith("|"):
            block.append(text)
            continue
        close()
        match = _HEADING.match(text)
        if match:
            level = len(match.group(1))
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = match.group(2)
    close()
    return tables


def report_table(relative: str, *headings: str, column: str) -> pd.DataFrame:
    """The one table that sits under headings containing every given text and has a column containing `column`."""
    hits = [
        table["frame"]
        for table in markdown_tables(relative)
        if all(any(h.lower() in found.lower() for found in table["headings"]) for h in headings)
        and any(column.lower() in name.lower() for name in table["frame"].columns)
    ]
    if len(hits) != 1:
        raise ReportError(f"{relative} does not have exactly one table under {' / '.join(headings)} with a '{column}' column")
    return hits[0].copy()


def report_bullets(relative: str, heading: str) -> list[str]:
    """The '- ' bullet points directly under the first heading that contains `heading`."""
    bullets: list[str] = []
    collecting, depth_of_section = False, 0
    for line in read_text(relative).splitlines():
        match = _HEADING.match(line.strip())
        if match:
            depth = len(match.group(1))
            if collecting and depth <= depth_of_section:
                break
            if not collecting and heading.lower() in match.group(2).lower():
                collecting, depth_of_section = True, depth
            continue
        if not collecting:
            continue
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.startswith("  ") and line.strip() and bullets:
            bullets[-1] += " " + line.strip()
    if not bullets:
        raise ReportError(f"{relative} has no bullet points under '{heading}'")
    return bullets


def leading_number(cell: str) -> str:
    """'0.615 [0.395, 0.832]' -> '0.615'."""
    match = re.match(r"-?\d+(?:\.\d+)?", cell)
    if not match:
        raise ReportError(f"'{cell}' does not start with a number")
    return match.group(0)


# --- display ----------------------------------------------------------------------------------------------
def show_table(frame: pd.DataFrame) -> None:
    """A static table that wraps long cells, so every column stays visible without sideways scrolling.

    The first column becomes the row label. (st.dataframe keeps each column wide and scrolls sideways inside its
    box, which hid the ROC-AUC column of the report tables on a laptop-sized window.)
    """
    st.table(frame.set_index(frame.columns[0]))


def show_figure(name: str, caption: str) -> None:
    """A saved figure from reports/figures/. Raises ReportError if it is missing."""
    path = ROOT / FIGURES / name
    if not path.is_file():
        raise ReportError(f"{FIGURES}/{name} is missing")
    st.image(str(path), caption=caption, width="stretch", alt=caption)
