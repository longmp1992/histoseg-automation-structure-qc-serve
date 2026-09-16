from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .core import (
    HEChangeDetectionConfig,
    HERegionSpec,
    HESegmentationConfig,
    run_he_change_detection,
    run_he_segmentation,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H&E region segmentation workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Extract tissue foreground or prompted regions.")
    _add_common_segmentation_args(single)
    single.add_argument(
        "--box",
        action="append",
        default=[],
        metavar="X0,Y0,X1,Y1",
        help="Prompt box for a target region. May be passed more than once.",
    )
    single.add_argument(
        "--point",
        action="append",
        default=[],
        metavar="X,Y",
        help="Positive prompt point for a target region. May be passed more than once.",
    )
    single.add_argument("--region-name", default="prompted_region", help="Name for prompted region output.")

    all_elements = subparsers.add_parser("all-elements", help="Partition H&E tissue into neutral components.")
    _add_common_segmentation_args(all_elements)
    all_elements.add_argument("--n-components", type=int, default=6, help="Number of neutral components.")
    all_elements.add_argument("--slic-segments", type=int, default=320, help="Approximate SLIC superpixel count.")
    all_elements.add_argument(
        "--component-name",
        action="append",
        default=[],
        help="Optional neutral component name. May be passed more than once.",
    )

    change = subparsers.add_parser("change", help="Detect changed regions between two aligned H&E images.")
    change.add_argument("--before", required=True, help="Before image path.")
    change.add_argument("--after", required=True, help="After image path.")
    change.add_argument("--out-dir", required=True, help="Output directory.")
    change.add_argument("--change-quantile", type=float, default=0.92, help="Change score quantile threshold.")
    change.add_argument("--min-change-area-px", type=int, default=256, help="Minimum change component area.")
    change.add_argument(
        "--resize-after-to-before",
        action="store_true",
        help="Resize after image if dimensions differ. No non-rigid registration is performed.",
    )

    args = parser.parse_args(argv)
    if args.command == "single":
        regions = []
        boxes = [_parse_float_list(value, expected=4, arg_name="--box") for value in args.box]
        points = [_parse_float_list(value, expected=2, arg_name="--point") for value in args.point]
        if boxes or points:
            regions.append(HERegionSpec(region_name=args.region_name, boxes=boxes, points=points))
        result = run_he_segmentation(
            HESegmentationConfig(
                image=args.image,
                out_dir=args.out_dir,
                task="single",
                backend=args.backend,
                medsam_model_id=args.medsam_model_id,
                region_specs=regions,
                min_region_area_px=args.min_region_area_px,
            )
        )
    elif args.command == "all-elements":
        result = run_he_segmentation(
            HESegmentationConfig(
                image=args.image,
                out_dir=args.out_dir,
                task="all_elements",
                backend=args.backend,
                medsam_model_id=args.medsam_model_id,
                n_components=args.n_components,
                component_names=args.component_name,
                slic_segments=args.slic_segments,
                min_region_area_px=args.min_region_area_px,
            )
        )
    else:
        result = run_he_change_detection(
            HEChangeDetectionConfig(
                before_image=args.before,
                after_image=args.after,
                out_dir=args.out_dir,
                resize_after_to_before=args.resize_after_to_before,
                change_quantile=args.change_quantile,
                min_change_area_px=args.min_change_area_px,
            )
        )

    print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))


def _add_common_segmentation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", required=True, help="Input H&E image path.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--backend",
        default="medsam",
        choices=["medsam", "heuristic"],
        help="Mask backend. Use heuristic for dependency-light local tests.",
    )
    parser.add_argument(
        "--medsam-model-id",
        default="wanglab/medsam-vit-base",
        help="Hugging Face model ID for backend=medsam.",
    )
    parser.add_argument("--min-region-area-px", type=int, default=256, help="Minimum output region area.")


def _parse_float_list(value: str, *, expected: int, arg_name: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected:
        raise SystemExit(f"{arg_name} expects {expected} comma-separated values, got {value!r}.")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise SystemExit(f"{arg_name} expects numeric comma-separated values, got {value!r}.") from exc


if __name__ == "__main__":
    main()
