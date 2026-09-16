"""
Convenience helpers for downloading raw files from a Hugging Face *dataset* repository.

We use `huggingface_hub.hf_hub_download` which:
  - downloads the file (or reuses cached version),
  - stores it in the local HF cache (version-aware),
  - returns the local filepath.

See HF docs for details on caching and repo_type="dataset".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from huggingface_hub import hf_hub_download


PathLike = Union[str, Path]


@dataclass(frozen=True)
class XeniumOutsPaths:
    repo_id: str
    revision: str
    cells_parquet: Path
    tissue_boundary_csv: Path
    clusters_csv: Path


def download_files(
    repo_id: str,
    filenames: Iterable[str],
    *,
    revision: str = "main",
    repo_type: str = "dataset",
    cache_dir: Optional[PathLike] = None,
) -> Dict[str, Path]:
    """Download multiple files from a HF repo and return local paths."""
    out: Dict[str, Path] = {}
    for fn in filenames:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fn,
            repo_type=repo_type,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        out[fn] = Path(local_path)
    return out


def download_xenium_outs(
    repo_id: str,
    *,
    revision: str = "main",
    clusters_relpath: str = "analysis/clustering/gene_expression_graphclust/clusters.csv",
    cache_dir: Optional[PathLike] = None,
) -> XeniumOutsPaths:
    """Download the 3 files needed by pattern1 isoline demo from a HF dataset repo."""
    files = download_files(
        repo_id=repo_id,
        filenames=["cells.parquet", "tissue_boundary.csv", clusters_relpath],
        revision=revision,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return XeniumOutsPaths(
        repo_id=repo_id,
        revision=revision,
        cells_parquet=files["cells.parquet"],
        tissue_boundary_csv=files["tissue_boundary.csv"],
        clusters_csv=files[clusters_relpath],
    )
