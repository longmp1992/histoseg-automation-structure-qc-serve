from __future__ import annotations

import argparse
import json

from .full_auto import (
    load_full_auto_spatial_pathologist_config,
    run_full_auto_spatial_pathologist,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full OpenAI-driven spatial pathologist workflow."
    )
    parser.add_argument("--config", required=True, help="Path to the full-auto workflow JSON.")
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Disable OpenAI calls and run the full workflow with heuristic annotation/review.",
    )
    args = parser.parse_args()

    cfg = load_full_auto_spatial_pathologist_config(args.config)
    if args.heuristic_only:
        cfg = type(cfg)(
            **{
                **cfg.__dict__,
                "openai_enabled": False,
            }
        )

    result = run_full_auto_spatial_pathologist(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
