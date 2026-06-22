"""Dependency-free writer for simple single-sheet ``.xlsx`` workbooks.

The project deliberately avoids a heavyweight Excel dependency for the
human-evaluation tooling. This module centralises the minimal OOXML scaffolding
so the annotation, sample, and adjudication workbooks share one implementation.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class XlsxDropdown:
    """A list (dropdown) data-validation applied to one named column."""

    column: str
    values: Sequence[str]
    allow_blank: bool = True


def _excel_column(index: int) -> str:
    column = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        column = chr(65 + remainder) + column
    return column


def _cell(row_index: int, column_index: int, value: Any) -> str:
    cell_ref = f"{_excel_column(column_index)}{row_index}"
    text = "" if value is None else str(value)
    return (
        f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">'
        f"{escape(text)}"
        "</t></is></c>"
    )


_WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

_PACKAGE_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""

_STYLES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""

_WORKBOOK_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="{sheet_name}" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""


def write_xlsx_sheet(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    dropdowns: Sequence[XlsxDropdown] = (),
    column_widths: Optional[Mapping[str, float]] = None,
    freeze_header: bool = True,
    sheet_name: str = "Sheet1",
) -> None:
    """Write ``rows`` under ``header`` to a single-sheet workbook at ``path``.

    ``dropdowns`` add list data-validations to named columns; ``column_widths``
    maps a header name to a column width. Cells are written as inline strings so
    no shared-strings table is required.
    """
    header = list(header)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = [header] + [list(row) for row in rows]
    max_row = max(2, len(all_rows))
    dimension = f"A1:{_excel_column(len(header))}{max_row}"

    sheet_rows = []
    for row_index, row in enumerate(all_rows, start=1):
        cells = "".join(
            _cell(row_index, column_index, value)
            for column_index, value in enumerate(row, start=1)
        )
        sheet_rows.append(f'<row r="{row_index}">{cells}</row>')

    cols_xml = ""
    if column_widths:
        col_entries = []
        for name, width in column_widths.items():
            if name not in header:
                continue
            index = header.index(name) + 1
            col_entries.append(
                f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
            )
        if col_entries:
            cols_xml = "<cols>" + "".join(col_entries) + "</cols>"

    panes_xml = (
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        "</sheetView></sheetViews>"
        if freeze_header
        else ""
    )

    validations = [dropdown for dropdown in dropdowns if dropdown.column in header]
    validations_xml = ""
    if validations:
        parts = []
        for dropdown in validations:
            col = _excel_column(header.index(dropdown.column) + 1)
            allow = "1" if dropdown.allow_blank else "0"
            formula = escape(",".join(str(value) for value in dropdown.values))
            parts.append(
                f'<dataValidation type="list" allowBlank="{allow}" '
                f'showErrorMessage="1" sqref="{col}2:{col}{max_row}">'
                f'<formula1>"{formula}"</formula1></dataValidation>'
            )
        validations_xml = (
            f'<dataValidations count="{len(parts)}">'
            + "".join(parts)
            + "</dataValidations>"
        )

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        f"{panes_xml}"
        '<sheetFormatPr defaultRowHeight="15"/>'
        f"{cols_xml}"
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        f"{validations_xml}"
        "</worksheet>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _PACKAGE_RELS)
        archive.writestr(
            "xl/workbook.xml", _WORKBOOK_XML.format(sheet_name=escape(sheet_name))
        )
        archive.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        archive.writestr("xl/styles.xml", _STYLES_XML)
