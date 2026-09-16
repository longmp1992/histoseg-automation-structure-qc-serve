from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import json


def _marker_list(markers: list[str]) -> str:
    if not markers:
        return "<span class='muted'>None</span>"
    return ", ".join(escape(marker) for marker in markers)


def write_annotation_report(
    *,
    output_dir: str | Path,
    evidence_pack: dict[str, Any],
    cluster_annotations: list[dict[str, Any]],
    case_review: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for annotation in sorted(cluster_annotations, key=lambda item: int(item["cluster_id"])):
        evidence = next(
            cluster for cluster in evidence_pack["clusters"] if int(cluster["cluster_id"]) == int(annotation["cluster_id"])
        )
        rows.append(
            f"""
            <tr>
              <td>{int(annotation["cluster_id"])}</td>
              <td>{escape(str(annotation["detailed_label"]))}</td>
              <td>{escape(str(annotation["broad_family"]))}</td>
              <td>{escape(str(annotation["malignancy_state"]))}</td>
              <td>{float(annotation["confidence"]):.3f}</td>
              <td>{escape(str(annotation["review_priority"]))}</td>
              <td>{int(evidence["cluster_size"])}</td>
              <td>{_marker_list(list(annotation.get("supporting_markers", [])))}</td>
              <td>{_marker_list([str(marker.get("gene")) for marker in evidence.get("top_positive_markers", [])[:8]])}</td>
              <td>{escape(str(annotation.get("reasoning_summary", "")))}</td>
            </tr>
            """
        )

    high_priority = ", ".join(str(cluster_id) for cluster_id in case_review.get("high_priority_cluster_ids", [])) or "None"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escape(str(evidence_pack["case_name"]))} Annotation Review</title>
  <style>
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f8fafc;
    }}
    h1, h2 {{
      margin-bottom: 0.35rem;
    }}
    p, li {{
      line-height: 1.5;
    }}
    .card {{
      background: white;
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 18px;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #eff6ff;
    }}
    .muted {{
      color: #6b7280;
    }}
    ul {{
      margin-top: 0.5rem;
      padding-left: 1.25rem;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{escape(str(evidence_pack["case_name"]))} Cluster Annotation Review</h1>
    <p>{escape(str(evidence_pack["study_context"]))}</p>
    <p><strong>Headline:</strong> {escape(str(case_review.get("headline", "")))}</p>
    <p><strong>Overall impression:</strong> {escape(str(case_review.get("overall_impression", "")))}</p>
    <p><strong>High-priority clusters:</strong> {escape(high_priority)}</p>
  </div>
  <div class="card">
    <h2>Consistency Notes</h2>
    <ul>
      {"".join(f"<li>{escape(str(note))}</li>" for note in case_review.get("consistency_notes", [])) or "<li class='muted'>No notes.</li>"}
    </ul>
    <h2>Discovery Candidates</h2>
    <p>{escape(', '.join(str(value) for value in case_review.get("discovery_candidates", [])) or 'None')}</p>
  </div>
  <div class="card">
    <h2>Cluster Table</h2>
    <table>
      <thead>
        <tr>
          <th>Cluster</th>
          <th>Detailed Label</th>
          <th>Family</th>
          <th>Malignancy</th>
          <th>Confidence</th>
          <th>Review</th>
          <th>Cells</th>
          <th>Supporting Markers</th>
          <th>Top Positive Markers</th>
          <th>Summary</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </div>
  <div class="card">
    <h2>Raw Case Review JSON</h2>
    <pre>{escape(json.dumps(case_review, indent=2, ensure_ascii=False))}</pre>
  </div>
</body>
</html>
"""
    report_path = output_dir / "annotation_review.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
