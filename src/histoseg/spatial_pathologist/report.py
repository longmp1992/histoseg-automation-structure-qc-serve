from __future__ import annotations

from pathlib import Path
from typing import Any
from html import escape
import json
import re


def _link(from_dir: Path, target: str) -> str:
    target_path = Path(target).resolve()
    try:
        return target_path.relative_to(from_dir).as_posix()
    except Exception:
        try:
            return Path("..", target_path.relative_to(from_dir.parent)).as_posix()
        except Exception:
            return target_path.as_posix()


def _sanitize_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and lines[-1].strip().lower().startswith(("if you want", "i can also", "i can convert")):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).strip()


def _format_inline_markdown(text: str) -> str:
    rendered = escape(str(text))
    rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"\*(.+?)\*", r"<em>\1</em>", rendered)
    return rendered


def _render_rich_text(text: str) -> str:
    sanitized = _sanitize_text(text)
    if not sanitized:
        return "<p class='muted'>None</p>"

    blocks: list[str] = []
    paragraph_lines: list[str] = []
    list_items: list[str] = []
    list_type: str | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip())
            if paragraph:
                blocks.append(f"<p>{_format_inline_markdown(paragraph)}</p>")
            paragraph_lines = []

    def flush_list() -> None:
        nonlocal list_items, list_type
        if list_items and list_type:
            tag = "ol" if list_type == "ol" else "ul"
            blocks.append(f"<{tag}>" + "".join(f"<li>{_format_inline_markdown(item)}</li>" for item in list_items) + f"</{tag}>")
        list_items = []
        list_type = None

    for raw_line in sanitized.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        if stripped in {"---", "***"}:
            flush_paragraph()
            flush_list()
            blocks.append("<hr class='section-sep'>")
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)

        if heading_match:
            flush_paragraph()
            flush_list()
            level = min(len(heading_match.group(1)) + 1, 4)
            blocks.append(f"<h{level}>{_format_inline_markdown(heading_match.group(2).strip())}</h{level}>")
            continue

        if ordered_match:
            flush_paragraph()
            if list_type not in {None, "ol"}:
                flush_list()
            list_type = "ol"
            list_items.append(ordered_match.group(1).strip())
            continue

        if bullet_match:
            flush_paragraph()
            if list_type not in {None, "ul"}:
                flush_list()
            list_type = "ul"
            list_items.append(bullet_match.group(1).strip())
            continue

        if list_type is not None:
            list_items[-1] = list_items[-1] + " " + stripped
        else:
            paragraph_lines.append(stripped)

    flush_paragraph()
    flush_list()
    return "".join(blocks)


def _render_list(items: list[str], *, rich: bool = False) -> str:
    if not items:
        return "<p class='muted'>None</p>"
    if rich:
        return "<ul>" + "".join(f"<li>{_render_rich_text(str(item))}</li>" for item in items) + "</ul>"
    return "<ul>" + "".join(f"<li>{escape(str(item))}</li>" for item in items) + "</ul>"


def _brief_text(text: str, *, max_chars: int = 360) -> str:
    sanitized = _sanitize_text(text)
    if not sanitized:
        return ""
    blocks = [block.strip() for block in sanitized.split("\n\n") if block.strip()]
    first_block = ""
    for block in blocks:
        single = " ".join(block.split())
        if re.fullmatch(r"\*{1,2}[^*]+\*{1,2}", single):
            continue
        if re.fullmatch(r"#{1,4}\s+.+", single):
            continue
        first_block = block
        break
    if not first_block and blocks:
        first_block = blocks[0]
    single_line = " ".join(first_block.split())
    single_line = re.sub(r"\*\*(.+?)\*\*", r"\1", single_line)
    single_line = re.sub(r"\*(.+?)\*", r"\1", single_line)
    if len(single_line) <= max_chars:
        return single_line
    cutoff = single_line[: max_chars + 1].rsplit(" ", 1)[0].strip()
    return cutoff + "..."


def _render_kv_table(mapping: dict[str, Any]) -> str:
    rows = []
    for key, value in mapping.items():
        rows.append(
            "<tr>"
            f"<th>{escape(str(key))}</th>"
            f"<td>{escape(str(value))}</td>"
            "</tr>"
        )
    return "<table class='kv-table'>" + "".join(rows) + "</table>"


def _render_media_card(*, href: str, title: str, description: str, alt: str) -> str:
    extension = Path(href).suffix.lower()
    if extension in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        media = f"<img src='{escape(href)}' alt='{escape(alt)}'>"
    else:
        media = (
            "<div class='file-card'>"
            f"<p class='muted'>Preview unavailable in-browser for {escape(extension or 'this file type')}.</p>"
            f"<p><a href='{escape(href)}' target='_blank' rel='noopener'>Open artifact</a></p>"
            "</div>"
        )
    return (
        "<article class='card image-card'>"
        f"{media}"
        f"<div class='body'><h3>{escape(title)}</h3><p class='muted'>{escape(description)}</p></div>"
        "</article>"
    )


def _render_structure_cards(structure_reviews: list[dict[str, Any]]) -> str:
    cards = []
    for review in structure_reviews:
        top_clusters = "".join(
            "<tr>"
            f"<td>C{int(cluster['cluster_id'])}</td>"
            f"<td>{escape(str(cluster['label']))}</td>"
            f"<td>{int(cluster['cell_count'])}</td>"
            f"<td>{100.0 * float(cluster['fraction_of_structure']):.1f}%</td>"
            "</tr>"
            for cluster in review.get("top_clusters", [])
        )
        cards.append(
            "<section class='card structure-card'>"
            f"<div class='card-header'><h3>{escape(review['title'])}</h3>"
            f"<span class='pill pill-{escape(review['review_priority'])}'>{escape(review['review_priority'])}</span></div>"
            f"{_render_rich_text(review['summary'])}"
            f"<p><strong>Confidence:</strong> {float(review['confidence']):.3f}</p>"
            f"<p><strong>Top cell types:</strong> {escape(review['top_celltype_summary'])}</p>"
            "<div class='subgrid'>"
            "<div>"
            "<h4>Key evidence</h4>"
            f"{_render_list([str(item) for item in review.get('key_evidence', [])], rich=True)}"
            "</div>"
            "<div>"
            "<h4>Recommended checks</h4>"
            f"{_render_list([str(item) for item in review.get('recommended_checks', [])], rich=True)}"
            "</div>"
            "</div>"
            "<h4>Top contributing clusters</h4>"
            "<table><thead><tr><th>Cluster</th><th>Label</th><th>Cells</th><th>Within structure</th></tr></thead>"
            f"<tbody>{top_clusters}</tbody></table>"
            "</section>"
        )
    return "".join(cards)


def _render_cluster_table(cluster_reviews: list[dict[str, Any]]) -> str:
    rows = []
    for review in cluster_reviews:
        rows.append(
            "<tr>"
            f"<td>C{int(review['cluster_id'])}</td>"
            f"<td>{escape(str(review['standardized_label']))}</td>"
            f"<td>{escape(str(review['cell_family']))}</td>"
            f"<td>{float(review['confidence']):.3f}</td>"
            f"<td>{escape(str(review['review_priority']))}</td>"
            f"<td>{escape(str(review['summary']))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Cluster</th><th>Label</th><th>Family</th><th>Confidence</th>"
        "<th>Priority</th><th>Summary</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def write_html_report(
    *,
    output_dir: Path,
    case_bundle: dict[str, Any],
    cluster_reviews: list[dict[str, Any]],
    structure_reviews: list[dict[str, Any]],
    case_summary: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    images = case_bundle["images"]
    report_path = output_dir / "index.html"
    report_json = output_dir / "case_bundle.json"
    report_json.write_text(
        json.dumps(
            {
                "case_bundle": case_bundle,
                "cluster_reviews": cluster_reviews,
                "structure_reviews": structure_reviews,
                "case_summary": case_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    he_overlay = _link(output_dir, images["he_overlay"])
    spatial_overlay = _link(output_dir, images["spatial_overlay"])
    clustermap = _link(output_dir, images["clustermap"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(case_bundle['case_name'])} | AI Spatial Pathologist</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffdfa;
      --ink: #1f2933;
      --muted: #66737f;
      --accent: #9a3412;
      --line: #e7dccd;
      --high: #b42318;
      --medium: #b54708;
      --low: #027a48;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(154, 52, 18, 0.08), transparent 24%),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px 64px; }}
    h1, h2, h3, h4 {{ margin: 0 0 12px; font-weight: 700; line-height: 1.15; }}
    h1 {{ font-size: 42px; letter-spacing: -0.03em; }}
    h2 {{ margin-top: 28px; font-size: 24px; }}
    p, li, td, th {{ line-height: 1.55; font-size: 15px; }}
    .hero, .card {{ background: rgba(255, 253, 250, 0.92); border: 1px solid var(--line); border-radius: 20px; box-shadow: 0 18px 40px rgba(71, 52, 35, 0.08); }}
    .hero {{ padding: 28px; }}
    .hero-grid, .metrics-grid, .image-grid, .structure-grid {{ display: grid; gap: 18px; }}
    .hero-grid {{ grid-template-columns: 2fr 1fr; align-items: start; }}
    .metrics-grid {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 20px; }}
    .metric {{ padding: 18px; background: #fff9f2; border: 1px solid var(--line); border-radius: 16px; }}
    .metric-value {{ display: block; font-size: 28px; margin-top: 8px; }}
    .image-grid {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin-top: 18px; }}
    .image-card {{ overflow: hidden; }}
    .image-card img {{ width: 100%; display: block; border-top-left-radius: 20px; border-top-right-radius: 20px; }}
    .file-card {{ padding: 28px 20px; min-height: 220px; display: flex; flex-direction: column; justify-content: center; align-items: flex-start; }}
    .image-card .body, .card-body {{ padding: 16px 18px 18px; }}
    .structure-grid {{ grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); margin-top: 18px; }}
    .structure-card {{ padding: 18px; }}
    .subgrid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
    .pill {{ display: inline-flex; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }}
    .pill-high {{ background: rgba(180, 35, 24, 0.12); color: var(--high); }}
    .pill-medium {{ background: rgba(181, 71, 8, 0.12); color: var(--medium); }}
    .pill-low {{ background: rgba(2, 122, 72, 0.12); color: var(--low); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 8px 10px; vertical-align: top; }}
    .kv-table th {{ width: 220px; color: var(--muted); font-weight: 600; }}
    .muted {{ color: var(--muted); }}
    .rich-text p {{ margin: 0 0 12px; }}
    .rich-text ul, .rich-text ol {{ margin: 0 0 14px 20px; padding: 0; }}
    .rich-text h3, .rich-text h4 {{ margin-top: 16px; }}
    .section-sep {{ border: 0; border-top: 1px solid var(--line); margin: 18px 0; }}
    a {{ color: var(--accent); text-decoration: none; }}
    @media (max-width: 900px) {{
      .hero-grid, .subgrid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 32px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <p class="muted">AI-Driven Spatial Pathologist</p>
          <h1>{escape(case_bundle['case_name'])}</h1>
          <p>{escape(case_summary['headline'])}</p>
          <p>{escape(_brief_text(case_summary['overall_impression']))}</p>
        </div>
        <div>
          {_render_kv_table({
              "Base pipeline config": case_bundle["base_pipeline_config"],
              "Analysis artifacts": case_bundle["analysis_output_dir"],
              "Validation artifacts": case_bundle["validation_output_dir"],
              "Review bundle": report_json.as_posix(),
          })}
        </div>
      </div>
      <div class="metrics-grid">
        <div class="metric"><span class="muted">Total cells</span><span class="metric-value">{int(case_bundle['total_cells'])}</span></div>
        <div class="metric"><span class="muted">Structures</span><span class="metric-value">{len(structure_reviews)}</span></div>
        <div class="metric"><span class="muted">Clusters</span><span class="metric-value">{len(cluster_reviews)}</span></div>
        <div class="metric"><span class="muted">High-priority reviews</span><span class="metric-value">{len(case_summary['review_priorities'])}</span></div>
      </div>
    </section>

    <section>
      <h2>Case overlays</h2>
      <div class="image-grid">
        {_render_media_card(href=he_overlay, title="H&E structure overlay", description="Spatial structures projected onto the aligned H&E image.", alt="H&E overlay")}
        {_render_media_card(href=spatial_overlay, title="Spatial structure overlay", description="Structure partitions in transcriptomic coordinate space.", alt="Spatial overlay")}
        {_render_media_card(href=clustermap, title="Structure clustermap", description="Cophenetic structure map from the rule-based structure assignment stage.", alt="Structure clustermap")}
      </div>
    </section>

    <section>
      <h2>Case synthesis</h2>
      <div class="card card-body">
        <h3>Overall interpretation</h3>
        <div class="rich-text">{_render_rich_text(case_summary['overall_impression'])}</div>
        <h3>Key findings</h3>
        {_render_list(case_summary['key_findings'], rich=True)}
        <h3>Review priorities</h3>
        {_render_list(case_summary['review_priorities'])}
        <h3>Discovery candidates</h3>
        {_render_list(case_summary['discovery_candidates'])}
      </div>
    </section>

    <section>
      <h2>Structure interpretations</h2>
      <div class="structure-grid">
        {_render_structure_cards(structure_reviews)}
      </div>
    </section>

    <section>
      <h2>Cluster review table</h2>
      <div class="card card-body">
        {_render_cluster_table(cluster_reviews)}
      </div>
    </section>
  </main>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path
