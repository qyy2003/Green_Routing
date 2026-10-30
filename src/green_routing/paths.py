"""Where things live: the repository checkout and the artifacts directory.

Everything that writes or reads run outputs (panel caches, run directories, the model
zoo, the viewer's default run) resolves them through here, so the checkout can sit
anywhere and the outputs can live on another disk:

    GR_ARTIFACTS_DIR   where run outputs go          (default: <repo>/artifacts)
    GR_REFERENCE_DIR   curated inputs that are not in the bucket: sites, external
                       links, DNS snapshots, the Zoo sample (default: <artifacts>/reference)
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

# src/green_routing -> the checkout root. This project is used as an editable install
# (`pip install -e .`), so the checkout is always there; for a plain install there is
# no checkout and the current directory stands in.
_checkout = PACKAGE_DIR.parents[1]
REPO_ROOT = _checkout if (_checkout / "pyproject.toml").exists() else Path.cwd()


def artifacts_dir() -> Path:
    """Run outputs: ``$GR_ARTIFACTS_DIR`` or ``<repo>/artifacts``."""
    raw = os.environ.get("GR_ARTIFACTS_DIR", "").strip()
    return Path(raw).expanduser().resolve() if raw else REPO_ROOT / "artifacts"


ARTIFACTS_DIR = artifacts_dir()


def reference_dir() -> Path:
    """Curated inputs of the network and carbon pipelines: ``$GR_REFERENCE_DIR`` or
    ``<artifacts>/reference``. Kept out of the repository (see docs/carbon.md)."""
    raw = os.environ.get("GR_REFERENCE_DIR", "").strip()
    return Path(raw).expanduser().resolve() if raw else artifacts_dir() / "reference"
