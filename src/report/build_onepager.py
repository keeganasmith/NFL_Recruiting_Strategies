"""Assemble a one-page NFL recruiting strategy report PDF with preflight checks."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

MIN_FONT_SIZE = 8
PAGE_SIZE = LETTER
MARGIN = 1 * inch


@dataclass(frozen=True)
class LayoutSpec:
    page_width: float = PAGE_SIZE[0]
    page_height: float = PAGE_SIZE[1]
    margin: float = MARGIN

    @property
    def content_width(self) -> float:
        return self.page_width - (2 * self.margin)


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _derive_data_cutoff_date(predictions_csv: Path | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if predictions_csv and predictions_csv.exists():
        preds = pd.read_csv(predictions_csv, usecols=["combine_year"])
        if not preds.empty and "combine_year" in preds.columns:
            max_year = int(preds["combine_year"].dropna().max())
            return f"{max_year}-12-31"
    return date.today().isoformat()


def _preflight_layout(
    layout: LayoutSpec, body_font_size: int, footer_font_size: int
) -> None:
    if layout.margin != MARGIN:
        raise AssertionError("Template margin must remain exactly one inch.")
    if body_font_size < MIN_FONT_SIZE or footer_font_size < MIN_FONT_SIZE:
        raise AssertionError(f"All text must be Arial at >= {MIN_FONT_SIZE} pt.")


def _merge_panel(
    base_page, panel_pdf: Path, x: float, y: float, width: float, height: float
) -> None:
    panel_reader = PdfReader(str(panel_pdf))
    panel = panel_reader.pages[0]
    panel_w = float(panel.mediabox.width)
    panel_h = float(panel.mediabox.height)
    scale = min(width / panel_w, height / panel_h)
    transform = Transformation().scale(scale, scale).translate(tx=x, ty=y)
    base_page.merge_transformed_page(panel, transform)


def _preflight_pdf(output_pdf: Path) -> None:
    reader = PdfReader(str(output_pdf))
    if len(reader.pages) != 1:
        raise AssertionError("Preflight failed: output must contain exactly one page.")
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    expected_width, expected_height = PAGE_SIZE
    if abs(width - expected_width) > 1 or abs(height - expected_height) > 1:
        raise AssertionError(
            "Preflight failed: page size must be US Letter (8.5 x 11)."
        )


def build_onepager(
    effect_panel_pdf: Path,
    diagnostics_panel_pdf: Path,
    metadata_json: Path,
    output_pdf: Path,
    predictions_csv: Path | None = None,
    data_cutoff_date: str | None = None,
) -> Path:
    _require_file(effect_panel_pdf, "Effect-size panel")
    _require_file(diagnostics_panel_pdf, "Diagnostics panel")
    _require_file(metadata_json, "Model metadata")
    if predictions_csv is not None:
        _require_file(predictions_csv, "Predictions CSV")

    metadata = _load_json(metadata_json)
    cutoff_date = _derive_data_cutoff_date(predictions_csv, data_cutoff_date)

    layout = LayoutSpec()
    body_font_size = 9
    footer_font_size = 10
    _preflight_layout(layout, body_font_size, footer_font_size)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        temp_base = Path(tmp.name)

    c = canvas.Canvas(str(temp_base), pagesize=PAGE_SIZE)

    top_y = layout.page_height - layout.margin - (3.5 * inch)
    bottom_y = layout.page_height - layout.margin - (6.25 * inch)

    c.setFont("Helvetica", body_font_size)
    lines = [
        "Prediction model: position group ridge regression on standardized combine metrics",
        "(speed, jumps, agility, size)",
        "NFL production value heuristic: weighted composite of starts, approximate value, snap share,",
        "and seasons played.",
        "If the heatmap legend did not come out correctly, red means positive impact on NFL production, blue means negative impact on NFL production",
    ]
    y = layout.margin + 94
    for line in lines:
        c.drawString(layout.margin, y, line)
        y -= 12

    c.showPage()
    c.save()

    base_reader = PdfReader(str(temp_base))
    base_page = base_reader.pages[0]
    # Merge the diagnostics panel first so the effect-size panel is merged last.
    # Plotly heatmap PDFs can carry gradient resources that are occasionally
    # flattened incorrectly when another PDF page is merged afterwards.
    _merge_panel(
        base_page,
        diagnostics_panel_pdf,
        x=0,
        y=bottom_y,
        width=layout.page_width,
        height=2.35 * inch,
    )
    _merge_panel(
        base_page,
        effect_panel_pdf,
        x=0,
        y=top_y,
        width=layout.page_width,
        height=3.25 * inch,
    )

    writer = PdfWriter()
    writer.add_page(base_page)
    with output_pdf.open("wb") as fh:
        writer.write(fh)

    temp_base.unlink(missing_ok=True)
    _preflight_pdf(output_pdf)
    return output_pdf


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--effect-panel-source",
        choices=("dotplot", "heatmap"),
        default="dotplot",
        help="Default effect-size panel type when --effect-panel-pdf is not provided.",
    )
    parser.add_argument(
        "--effect-panel-pdf",
        type=Path,
        default=None,
        help="Optional explicit effect-size panel PDF path.",
    )
    parser.add_argument(
        "--diagnostics-panel-pdf",
        type=Path,
        default=Path("outputs/visualizations/model_diagnostics.pdf"),
    )
    parser.add_argument(
        "--metadata-json", type=Path, default=Path("outputs/modeling/metadata.json")
    )
    parser.add_argument(
        "--predictions-csv", type=Path, default=Path("outputs/modeling/predictions.csv")
    )
    parser.add_argument("--data-cutoff-date", default=None)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("output/NFL_recruiting_strategy_onepager.pdf"),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    default_effect_panel = Path(
        f"outputs/visualizations/effect_size_{args.effect_panel_source}.pdf"
    )
    build_onepager(
        effect_panel_pdf=args.effect_panel_pdf or default_effect_panel,
        diagnostics_panel_pdf=args.diagnostics_panel_pdf,
        metadata_json=args.metadata_json,
        predictions_csv=args.predictions_csv,
        data_cutoff_date=args.data_cutoff_date,
        output_pdf=args.output_pdf,
    )
    print(f"Built one-pager: {args.output_pdf}")


if __name__ == "__main__":
    main()
