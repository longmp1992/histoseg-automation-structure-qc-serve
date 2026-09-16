from __future__ import annotations

import argparse
import json

from .config import load_spatial_pathologist_config
from .runner import run_spatial_pathologist


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AI-driven spatial pathologist product workflow."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the spatial pathologist case config JSON.",
    )
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Disable OpenAI calls even if the API key is available.",
    )
    args = parser.parse_args()

    cfg = load_spatial_pathologist_config(args.config)
    if args.heuristic_only:
        cfg = type(cfg)(**{**cfg.__dict__, "openai_enabled": False})

    result = run_spatial_pathologist(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
