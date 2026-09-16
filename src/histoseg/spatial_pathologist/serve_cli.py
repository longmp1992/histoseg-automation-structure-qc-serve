from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence


class _NullProgress:
    def __call__(self, *args: object, **kwargs: object) -> None:
        return None


def _split_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _parse_group_ids(raw: str | None) -> set[int] | str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.lower() == "all":
        return "all"

    group_ids: set[int] = set()
    for token in _split_csv_tokens(value):
        try:
            group_id = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--select-group-ids must be a comma-separated list of integers, or 'all'."
            ) from exc
        if group_id <= 0:
            raise argparse.ArgumentTypeError("--select-group-ids values must be positive integers.")
        group_ids.add(group_id)
    return group_ids


def _read_structure_lines(args: argparse.Namespace) -> list[str]:
    lines: list[str] = []
    if args.structures_file:
        path = Path(args.structures_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Structure file does not exist: {path}")
        lines.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines())

    for item in args.structures or []:
        value = str(item).strip()
        if value:
            lines.append(value)

    return [line for line in lines if line and not line.lstrip().startswith("#")]


def _select_groups_from_state(group_state: dict[str, object], selector: set[int] | str | None) -> list[str]:
    if selector is None:
        return []
    choices = [str(choice) for choice in group_state.get("choices", [])]
    records = [dict(record) for record in group_state.get("group_records", [])]
    if selector == "all":
        return choices

    selected: list[str] = []
    available: dict[int, str] = {}
    for record in records:
        group_id = int(record["group_id"])
        label = str(record["choice_label"])
        available[group_id] = label
        if group_id in selector:
            selected.append(label)

    missing = sorted(selector.difference(available))
    if missing:
        available_text = ", ".join(str(item) for item in sorted(available))
        raise ValueError(
            f"Unknown structure group id(s): {', '.join(str(item) for item in missing)}. "
            f"Available ids: {available_text}"
        )
    return selected


def _copy_file(source: str | Path | None, out_dir: Path, *, name: str | None = None) -> Path | None:
    if source is None:
        return None
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        return None
    destination = out_dir / (name or source_path.name)
    if source_path == destination.resolve():
        return destination
    shutil.copy2(source_path, destination)
    return destination


def _copy_many(paths: Iterable[str], out_dir: Path) -> list[str]:
    copied: list[str] = []
    seen: set[Path] = set()
    for raw_path in paths:
        source = Path(raw_path).expanduser().resolve()
        if source in seen or not source.exists():
            continue
        seen.add(source)
        copied_path = _copy_file(source, out_dir)
        if copied_path is not None:
            copied.append(str(copied_path))
    return copied


def _write_selection_artifacts(
    *,
    status: str,
    selector_path: str | None,
    heatmap_path: str | None,
    group_table: object,
    group_state: dict[str, object],
    out_dir: Path,
) -> None:
    (out_dir / "dendrogram_status.txt").write_text(status, encoding="utf-8")
    (out_dir / "structure_group_state.json").write_text(
        json.dumps(group_state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _copy_file(selector_path, out_dir, name="interactive_structure_selector_none.png")
    _copy_file(heatmap_path, out_dir, name="cophenetic_heatmap_all_clusters.png")
    if hasattr(group_table, "to_csv"):
        group_table.to_csv(out_dir / "candidate_structure_groups.csv", index=False)


def _last_generator_value(values: Iterable[object]) -> object:
    sentinel = object()
    last: object = sentinel
    for last in values:
        pass
    if last is sentinel:
        raise RuntimeError("The Serve app run did not produce any output.")
    return last


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the AI Driven Spatial Pathologist Serve workflow from the command line "
            "without launching Gradio."
        )
    )
    parser.add_argument("--cells-parquet", required=True, help="Path to Xenium cells.parquet.")
    parser.add_argument("--clusters-csv", required=True, help="Path to GraphClust clusters.csv.")
    parser.add_argument("--tissue-boundary-csv", default=None, help="Optional tissue_boundary.csv path.")
    parser.add_argument("--out-dir", required=True, help="Directory where website-equivalent outputs are copied.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Intermediate Serve workspace. Defaults to <out-dir>/_serve_work.",
    )
    parser.add_argument(
        "--structures",
        nargs="*",
        default=None,
        help=(
            "Manual structure lines, one argument per structure, e.g. "
            "--structures \"10,23,19\" \"27,14,20\". This matches the website text box."
        ),
    )
    parser.add_argument(
        "--structures-file",
        default=None,
        help="Text file with one comma-separated structure per line. Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--n-structure-groups",
        type=int,
        default=4,
        help="Dendrogram cut count used by the website Build step.",
    )
    parser.add_argument(
        "--select-group-ids",
        default=None,
        help=(
            "Comma-separated dendrogram structure ids to select, e.g. 1,3, or 'all'. "
            "Use this when you want the CLI to mimic checklist selection."
        ),
    )
    parser.add_argument(
        "--list-groups-only",
        action="store_true",
        help="Only run the dendrogram Build step and write candidate structure artifacts.",
    )
    parser.add_argument("--grid-n", type=int, default=900, help="Partition grid bins along X.")
    parser.add_argument("--knn-k", type=int, default=700, help="Partition grid bins along Y.")
    parser.add_argument("--smooth-sigma", type=float, default=2.25, help="Gaussian smoothing sigma.")
    parser.add_argument(
        "--min-cells-inside",
        type=int,
        default=500,
        help="Minimum cells required for one structure.",
    )
    parser.add_argument(
        "--min-dominance",
        type=float,
        default=0.34,
        help="Minimum dominance needed to claim a pixel.",
    )
    parser.add_argument(
        "--support-quantile",
        type=float,
        default=0.18,
        help="Support threshold quantile.",
    )
    parser.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="Match the website checkbox being turned off.",
    )
    parser.add_argument(
        "--no-cophenetic-heatmap",
        action="store_true",
        help="Skip the final cophenetic heatmap output.",
    )
    parser.add_argument(
        "--save-selection-artifacts",
        action="store_true",
        help="Also copy dendrogram Build-step artifacts into out-dir.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cells_parquet = Path(args.cells_parquet).expanduser().resolve()
    clusters_csv = Path(args.clusters_csv).expanduser().resolve()
    tissue_boundary_csv = (
        Path(args.tissue_boundary_csv).expanduser().resolve() if args.tissue_boundary_csv else None
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else out_dir / "_serve_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    for required_path in (cells_parquet, clusters_csv):
        if not required_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {required_path}")
    if tissue_boundary_csv is not None and not tissue_boundary_csv.exists():
        raise FileNotFoundError(f"Input file does not exist: {tissue_boundary_csv}")

    os.environ["APP_DATA_DIR"] = str(work_dir)

    from . import serve_app

    progress = _NullProgress()
    try:
        selector = _parse_group_ids(args.select_group_ids)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    structure_lines = _read_structure_lines(args)
    needs_build_step = args.list_groups_only or selector is not None or args.save_selection_artifacts

    group_state: dict[str, object] | None = None
    selected_groups: list[str] = []
    if needs_build_step:
        (
            build_status,
            selector_path,
            build_heatmap_path,
            group_table,
            _selector_update,
            _pattern_text,
            _selection_summary,
            group_state,
            _auto_lines,
        ) = serve_app.build_structure_groups(
            str(cells_parquet),
            str(clusters_csv),
            int(args.n_structure_groups),
            progress=progress,
        )
        if args.save_selection_artifacts or args.list_groups_only:
            _write_selection_artifacts(
                status=str(build_status),
                selector_path=str(selector_path) if selector_path else None,
                heatmap_path=str(build_heatmap_path) if build_heatmap_path else None,
                group_table=group_table,
                group_state=group_state,
                out_dir=out_dir,
            )
        selected_groups = _select_groups_from_state(group_state, selector)

    if args.list_groups_only:
        print(f"Wrote dendrogram candidate structure artifacts to: {out_dir}")
        return

    if not structure_lines and not selected_groups:
        raise SystemExit(
            "No structures selected. Provide --structures/--structures-file, or use "
            "--select-group-ids with ids from candidate_structure_groups.csv."
        )

    pattern1_clusters = "\n".join(structure_lines)
    final_output = _last_generator_value(
        serve_app.run_analysis(
            str(cells_parquet),
            str(clusters_csv),
            str(tissue_boundary_csv) if tissue_boundary_csv is not None else None,
            pattern1_clusters,
            selected_groups,
            group_state,
            int(args.grid_n),
            int(args.knn_k),
            float(args.smooth_sigma),
            int(args.min_cells_inside),
            not bool(args.no_cophenetic_heatmap),
            float(args.min_dominance),
            float(args.support_quantile),
            not bool(args.no_fill_holes),
            progress=progress,
        )
    )

    status_text, preview_path, heatmap_path, summary, archive_path, output_files = final_output
    copied_files = _copy_many(output_files, out_dir)
    copied_archive = _copy_file(archive_path, out_dir, name="histoseg_outputs.zip")
    _copy_file(preview_path, out_dir)
    _copy_file(heatmap_path, out_dir)

    (out_dir / "run_status.txt").write_text(str(status_text), encoding="utf-8")
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"Wrote website-equivalent output files to: {out_dir}")
    print(f"Copied {len(copied_files)} downloadable output file(s).")
    if copied_archive is not None:
        print(f"Copied output archive: {copied_archive}")


if __name__ == "__main__":
    main()
