"""Where the web/ builders read the Cricket CSVs from.

The same source as the green_routing package: the S3 bucket (each object downloaded
once into the green_routing object cache, then read from local disk) or a local
mirror (``GR_DATA_SOURCE=local`` + ``GR_LOCAL_ROOT``). Without the package installed
the builders fall back to ``GR_LOCAL_ROOT`` (default ``~/switch``).

All paths here are relative to the data root, e.g. ``router-interfaces/ag1/x.csv``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:                                    # the package: S3 + cache, or a local mirror
    from green_routing.dataset import config as _cfg, storage as _st
except ImportError:                     # stdlib-only fallback: a local mirror
    _cfg = _st = None

HERE = Path(__file__).resolve().parent


def _root() -> str:
    if _cfg is not None:
        return _cfg.DATA_ROOT.rstrip("/")
    return str(Path(os.environ.get("GR_LOCAL_ROOT", "~/switch")).expanduser())


def _using_s3() -> bool:
    return _cfg is not None and _cfg.USING_S3


def describe() -> str:
    return _cfg.describe_source() if _cfg is not None else f"local:{_root()}"


def topology_txt() -> Path:
    """The router topology: the research checkout's copy when present, else the copy
    committed with the package (the topology is not in the bucket)."""
    if _cfg is not None:
        return Path(_cfg.TOPOLOGY_TXT)
    return HERE.parent / "src" / "green_routing" / "dataset" / "switch-network-topology.txt"


@lru_cache(maxsize=None)
def _listing(subtree: str) -> frozenset[str]:
    """Relative paths of every CSV under ``subtree`` (one listing, cached)."""
    base = f"{_root()}/{subtree}"
    if _using_s3():
        n = len(_root()) + 1
        return frozenset(u[n:] for u in _st.walk_files(base))
    p = Path(base)
    if not p.exists():
        return frozenset()
    root = Path(_root())
    return frozenset(str(f.relative_to(root)) for f in p.rglob("*.csv"))


def exists(rel: str) -> bool:
    return rel in _listing(rel.split("/", 1)[0])


def names(subtree: str) -> list[str]:
    """CSV file names directly inside ``subtree`` (e.g. ``routers``)."""
    head = subtree.rstrip("/") + "/"
    return sorted(r[len(head):] for r in _listing(subtree.split("/", 1)[0])
                  if r.startswith(head) and "/" not in r[len(head):])


def local_path(rel: str) -> Path:
    """A local file to open(): the mirror's file, or the cached copy of the object."""
    if _using_s3():
        return Path(_st.local_copy(f"{_root()}/{rel}"))
    return Path(_root()) / rel
