"""Uniform file access over the local filesystem **and** S3 object storage.

The SWITCH Cricket time-series live in SWITCHengines object storage (bucket
``network-energy-efficiency-research``, written daily by ``parse_rrd.py``); the local
``~/switch`` tree is only ever a mirror of it. This module lets the rest of the
package address either one through the same handful of calls, so ``dataset`` runs
unchanged on a laptop with the disk mirror or on a cloud box with nothing but
credentials.

Everything is addressed by **URI string**:

    ~/switch/router-interfaces/ag1/te0_0_1_0.csv              local
    s3://network-energy-efficiency-research/router-interfaces/ag1/...  object storage

``is_s3`` dispatches; local URIs fall through to plain ``pathlib``/``open``, so a
path stored in an old ``cohort.csv`` keeps working after the switch to S3.

Object reads go through a **local disk cache** (``config.CACHE_DIR``): the first
read downloads, later reads hit the disk. Freshness is checked against the ETag
that ``list_objects_v2`` already returns while listing a prefix, so validation
costs no extra requests. Set ``GR_S3_CACHE=0`` to stream instead of caching.

Needs ``boto3`` for the S3 side (``pip install boto3``); the local side has no
dependencies beyond the stdlib.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Iterator

from . import config

S3_SCHEME = "s3://"


# --- URI helpers -------------------------------------------------------------

def is_s3(uri: str | Path) -> bool:
    return str(uri).startswith(S3_SCHEME)


def split_s3(uri: str) -> tuple[str, str]:
    """``s3://bucket/a/b.csv`` -> ``("bucket", "a/b.csv")``."""
    if not is_s3(uri):
        raise ValueError(f"not an s3 URI: {uri}")
    rest = str(uri)[len(S3_SCHEME):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def join(base: str | Path, *parts: str) -> str:
    """Join path components under a local path or an ``s3://`` URI."""
    if is_s3(base):
        head = str(base).rstrip("/")
        tail = "/".join(p.strip("/") for p in parts if p)
        return f"{head}/{tail}" if tail else head
    return str(Path(base).joinpath(*parts))


def basename(uri: str | Path) -> str:
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def stem(uri: str | Path) -> str:
    """Filename without its final suffix (``.../te0_0_1_0.csv`` -> ``te0_0_1_0``)."""
    name = basename(uri)
    return name.rsplit(".", 1)[0] if "." in name else name


def parent(uri: str | Path) -> str:
    s = str(uri).rstrip("/")
    if is_s3(s):
        head, sep, _ = s.rpartition("/")
        return head if sep and head != S3_SCHEME.rstrip("/") else s
    return str(Path(s).parent)


# --- S3 client ---------------------------------------------------------------

_client_lock = threading.Lock()
_client = None


def s3_client():
    """Process-wide boto3 S3 client built from ``config`` (endpoint/profile/region).

    boto3 clients are thread-safe for calls, so one shared client serves the
    parallel prefetch pool.
    """
    global _client
    with _client_lock:
        if _client is None:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
            except ImportError as exc:   # pragma: no cover - environment issue
                raise ImportError(
                    "S3 access needs boto3 — `pip install boto3`. Or set "
                    "GR_DATA_SOURCE=local to read a filesystem mirror instead."
                ) from exc
            session = boto3.Session(profile_name=config.S3_PROFILE)
            _client = session.client(
                "s3",
                endpoint_url=config.S3_ENDPOINT or None,
                region_name=config.S3_REGION or None,
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": config.S3_ADDRESSING},
                    retries={"max_attempts": 5, "mode": "standard"},
                    max_pool_connections=max(32, config.S3_WORKERS * 4),
                ),
            )
        return _client


_transfer_cfg = None


def _transfer_config():
    global _transfer_cfg
    if _transfer_cfg is None:
        from boto3.s3.transfer import TransferConfig
        _transfer_cfg = TransferConfig(use_threads=False)
    return _transfer_cfg


def reset_client() -> None:
    """Drop the cached client (after changing credentials/endpoint at runtime)."""
    global _client
    with _client_lock:
        _client = None


# --- Listing -----------------------------------------------------------------

@dataclass(frozen=True)
class ObjectInfo:
    """One stored object: URI plus the metadata a listing hands us for free."""
    uri: str
    size: int
    etag: str = ""
    last_modified: float = 0.0


_listing_lock = threading.Lock()
_listing_mem: dict[str, list[ObjectInfo]] = {}
# URI -> metadata for everything any listing has returned this process. Reads consult
# it to validate their cached copy without paying for a HEAD (see _resolve_info).
_info_index: dict[str, ObjectInfo] = {}


def _listing_cache_file(prefix_uri: str) -> Path:
    safe = prefix_uri.replace("://", "_").replace("/", "_").strip("_")
    return config.CACHE_DIR / "_listings" / f"{safe}.json"


def _load_listing_cache(prefix_uri: str) -> list[ObjectInfo] | None:
    if not config.S3_CACHE:         # cache off means no disk footprint at all
        return None
    fp = _listing_cache_file(prefix_uri)
    if not fp.exists():
        return None
    if config.S3_LIST_TTL and time.time() - fp.stat().st_mtime > config.S3_LIST_TTL:
        return None
    try:
        raw = json.loads(fp.read_text())
    except (OSError, ValueError):
        return None
    return [ObjectInfo(**o) for o in raw]


def _store_listing_cache(prefix_uri: str, objects: list[ObjectInfo]) -> None:
    if not config.S3_CACHE:
        return
    fp = _listing_cache_file(prefix_uri)
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(".tmp")
        tmp.write_text(json.dumps([o.__dict__ for o in objects]))
        tmp.replace(fp)
    except OSError:
        pass        # a cold listing every run is slow, not wrong


def _s3_list(prefix_uri: str) -> list[ObjectInfo]:
    bucket, key_prefix = split_s3(prefix_uri)
    key_prefix = key_prefix.rstrip("/") + "/" if key_prefix else ""
    client = s3_client()
    out: list[ObjectInfo] = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=key_prefix
    ):
        for obj in page.get("Contents", ()):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            lm = obj.get("LastModified")
            out.append(
                ObjectInfo(
                    uri=f"{S3_SCHEME}{bucket}/{key}",
                    size=int(obj.get("Size", 0)),
                    etag=str(obj.get("ETag", "")).strip('"'),
                    last_modified=lm.timestamp() if lm is not None else 0.0,
                )
            )
    return out


def walk_objects(prefix_uri: str, suffix: str | None = None) -> list[ObjectInfo]:
    """Every object under ``prefix_uri``, recursively, sorted by URI.

    One paginated ``list_objects_v2`` covers a whole subtree on S3 — far cheaper
    than per-directory listings — and the result is memoised in process and on
    disk (``config.S3_LIST_TTL``, default 12 h; the upstream pipeline writes
    daily). Local roots are walked with ``os.scandir``.
    """
    with _listing_lock:
        cached = _listing_mem.get(prefix_uri)
    if cached is None and is_s3(prefix_uri):
        cached = _load_listing_cache(prefix_uri)
    if cached is None:
        if is_s3(prefix_uri):
            cached = _s3_list(prefix_uri)
            _store_listing_cache(prefix_uri, cached)
        else:
            cached = _local_walk(prefix_uri)
        cached.sort(key=lambda o: o.uri)
        with _listing_lock:
            _listing_mem[prefix_uri] = cached
            _info_index.update({o.uri: o for o in cached})
    if suffix is None:
        return list(cached)
    return [o for o in cached if o.uri.endswith(suffix)]


def _local_walk(root: str) -> list[ObjectInfo]:
    base = Path(root)
    if not base.exists():
        return []
    out: list[ObjectInfo] = []
    for dirpath, _dirnames, filenames in os.walk(base):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                st = fp.stat()
            except OSError:
                continue
            out.append(ObjectInfo(uri=str(fp), size=st.st_size,
                                  last_modified=st.st_mtime))
    return out


def walk_files(prefix_uri: str, suffix: str = ".csv") -> list[str]:
    """URIs of every file under ``prefix_uri`` ending in ``suffix``.

    The ``.csv`` default also drops the stray ``*.csv.<hex>`` partials the
    upstream export leaves behind.
    """
    return [o.uri for o in walk_objects(prefix_uri, suffix=suffix)]


def list_dirs(prefix_uri: str) -> list[str]:
    """Immediate child "directories" of ``prefix_uri`` (device dirs, mostly)."""
    if is_s3(prefix_uri):
        head = str(prefix_uri).rstrip("/") + "/"
        names = {
            o.uri[len(head):].split("/", 1)[0]
            for o in walk_objects(prefix_uri)
            if o.uri.startswith(head) and "/" in o.uri[len(head):]
        }
        return sorted(join(prefix_uri, n) for n in names)
    base = Path(prefix_uri)
    if not base.exists():
        return []
    return sorted(str(p) for p in base.iterdir() if p.is_dir())


def exists(uri: str | Path) -> bool:
    if not is_s3(uri):
        return Path(uri).exists()
    bucket, key = split_s3(str(uri))
    try:
        s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# --- Object cache + reads ----------------------------------------------------

def cache_path(uri: str) -> Path:
    bucket, key = split_s3(uri)
    return config.CACHE_DIR / bucket / key


def _sidecar(fp: Path) -> Path:
    return fp.with_name(fp.name + ".meta")


def _cache_is_fresh(fp: Path, info: ObjectInfo | None) -> bool:
    """Cached copy usable? With listing metadata this is an ETag/size compare;
    without it we trust whatever is on disk (no extra HEAD per file)."""
    if not fp.exists():
        return False
    if info is None:
        return True
    if info.size and fp.stat().st_size != info.size:
        return False
    if not info.etag:
        return True
    try:
        return json.loads(_sidecar(fp).read_text()).get("etag") == info.etag
    except (OSError, ValueError):
        return False


def _resolve_info(uri: str) -> ObjectInfo | None:
    """Metadata for ``uri``, so a cached copy can be checked against the bucket.

    Free when a listing already covered the object (the scans list a whole subtree
    up front). For a one-off read of an unlisted URI it costs one HEAD -- worth it,
    because otherwise a stale cache entry would be served forever. ``GR_S3_VALIDATE=0``
    drops the HEAD and trusts whatever is cached.
    """
    with _listing_lock:
        info = _info_index.get(uri)
    if info is not None or not config.S3_VALIDATE:
        return info
    bucket, key = split_s3(uri)
    try:
        head = s3_client().head_object(Bucket=bucket, Key=key)
    except Exception:
        return None             # let the actual read raise with real context
    lm = head.get("LastModified")
    info = ObjectInfo(
        uri=uri,
        size=int(head.get("ContentLength", 0)),
        etag=str(head.get("ETag", "")).strip('"'),
        last_modified=lm.timestamp() if lm is not None else 0.0,
    )
    with _listing_lock:
        _info_index[uri] = info
    return info


def _download(uri: str, info: ObjectInfo | None = None) -> Path:
    """Fetch ``uri`` into the disk cache (no-op when the copy is current)."""
    fp = cache_path(uri)
    if _cache_is_fresh(fp, info):
        return fp
    bucket, key = split_s3(uri)
    fp.parent.mkdir(parents=True, exist_ok=True)
    # Download to a per-thread temp name, then rename: concurrent prefetchers and
    # readers never observe a half-written CSV.
    tmp = fp.with_name(f"{fp.name}.{os.getpid()}.{threading.get_ident()}.part")
    try:
        # use_threads=False: prefetch() already runs one thread per file, and letting
        # S3Transfer fan out again would oversubscribe the connection pool.
        s3_client().download_file(bucket, key, str(tmp), Config=_transfer_config())
        tmp.replace(fp)
    finally:
        tmp.unlink(missing_ok=True)
    if info is not None and info.etag:
        try:
            _sidecar(fp).write_text(json.dumps({"etag": info.etag, "size": info.size}))
        except OSError:
            pass
    return fp


def local_copy(uri: str, info: ObjectInfo | None = None) -> str:
    """A real filesystem path for ``uri`` — the file itself, or its cached copy.

    Only meaningful with caching on; raises for an S3 URI when ``GR_S3_CACHE=0``.
    """
    if not is_s3(uri):
        return str(uri)
    if not config.S3_CACHE:
        raise RuntimeError(f"{uri}: no local copy with GR_S3_CACHE=0")
    return str(_download(uri, info if info is not None else _resolve_info(uri)))


def open_binary(uri: str | Path, info: ObjectInfo | None = None) -> IO[bytes]:
    """Binary handle on a local path or an S3 object (cached, else streamed)."""
    uri = str(uri)
    if not is_s3(uri):
        return open(uri, "rb")
    if config.S3_CACHE:
        if info is None:
            info = _resolve_info(uri)
        return open(_download(uri, info), "rb")
    bucket, key = split_s3(uri)
    body = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return io.BytesIO(body)


def open_text(uri: str | Path, encoding: str = "utf-8", newline: str | None = None):
    """Text handle on a local path or an S3 object."""
    uri = str(uri)
    if not is_s3(uri):
        return open(uri, "r", encoding=encoding, newline=newline)
    return io.TextIOWrapper(open_binary(uri), encoding=encoding, newline=newline)


def read_bytes(uri: str | Path) -> bytes:
    with open_binary(uri) as fh:
        return fh.read()


def read_text(uri: str | Path, encoding: str = "utf-8") -> str:
    return read_bytes(uri).decode(encoding)


# --- Bulk prefetch -----------------------------------------------------------

def prefetch(
    objects: Iterable[ObjectInfo | str],
    workers: int | None = None,
    verbose: bool = False,
) -> int:
    """Warm the disk cache for many objects in parallel; returns the count fetched.

    A full coverage scan touches ~2 300 CSVs; pulled one at a time the round-trips
    dominate, so callers that are about to read a whole subtree should prefetch it
    first. Local URIs and an already-current cache are skipped, making repeat runs
    cheap. Failures are left for the individual read to surface.
    """
    if not config.S3_CACHE:
        return 0
    todo = [o for o in objects if is_s3(o.uri if isinstance(o, ObjectInfo) else o)]
    if not todo:
        return 0
    workers = workers or config.S3_WORKERS
    done = 0

    def _one(item: ObjectInfo | str) -> bool:
        info = item if isinstance(item, ObjectInfo) else None
        uri = item.uri if isinstance(item, ObjectInfo) else item
        if _cache_is_fresh(cache_path(uri), info):
            return False
        try:
            _download(uri, info)
            return True
        except Exception:
            return False        # let the read report it with proper context

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, fetched in enumerate(pool.map(_one, todo), 1):
            done += bool(fetched)
            if verbose and i % 200 == 0:
                print(f"      prefetch {i}/{len(todo)} ({done} downloaded)")
    if verbose:
        print(f"      prefetch done: {done} downloaded, "
              f"{len(todo) - done} already cached")
    return done


def clear_cache(prefix_uri: str | None = None) -> None:
    """Delete cached objects (all of them, or just those under ``prefix_uri``)."""
    with _listing_lock:
        _listing_mem.clear()
        _info_index.clear()
    target = config.CACHE_DIR if prefix_uri is None else Path(cache_path(prefix_uri))
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def cache_size_bytes() -> int:
    return sum(f.stat().st_size for f in config.CACHE_DIR.rglob("*") if f.is_file())


# --- Self-check --------------------------------------------------------------

def check(sample: int = 3, verbose: bool = True) -> dict:
    """Verify the configured source is reachable and readable.

    Resolves the source, lists the ``router-interfaces`` subtree, and reads one CSV
    end-to-end. Run this first on a new box -- it separates "credentials/endpoint
    wrong" from "the pipeline is broken".
    """
    from . import loaders

    report: dict = {"source": config.describe_source(), "ok": False}
    if verbose:
        print(f"source        : {report['source']}")
    root = config.SUBTREES["router-interfaces"]
    if config.USING_S3:
        # One live request: the listing below and the sampled CSVs may all come from
        # the local cache, which would hide wrong credentials on a warm machine.
        bucket, key = split_s3(root)
        try:
            s3_client().list_objects_v2(Bucket=bucket, Prefix=key.rstrip("/") + "/",
                                        MaxKeys=1)
        except Exception as exc:
            report["error"] = f"live request to the bucket failed: {exc!r}"
            if verbose:
                print(f"ERROR         : {report['error']}")
                print("                check ~/.aws/credentials, GR_S3_ENDPOINT and "
                      "GR_S3_BUCKET (docs/setup.md)")
            return report
        if verbose:
            print("credentials   : OK (live request to the bucket)")
    objs = walk_objects(root, suffix=".csv")
    series = list(loaders.iter_interface_files())
    report["n_interface_csv"] = len(series)
    report["n_devices"] = len({dev for dev, _, _ in series})
    report["total_bytes"] = sum(o.size for o in objs)
    if verbose:
        print(f"interface CSVs: {len(series)} on {report['n_devices']} devices "
              f"({report['total_bytes'] / 1e9:.1f} GB under {root})")
    if not series:
        report["error"] = f"no <device>/<iface>.csv under {root}"
        if verbose:
            print(f"ERROR         : {report['error']}")
        return report

    samples = []
    for dev, iface, uri in series[:sample]:
        df = loaders.interface_throughput(uri)
        samples.append(
            {
                "key": f"{dev}/{iface}",
                "rows": int(len(df)),
                "first": str(df.index.min()),
                "last": str(df.index.max()),
            }
        )
        if verbose:
            print(f"  read {dev}/{iface}: {len(df)} rows, "
                  f"{df.index.min()} .. {df.index.max()}")
    report["samples"] = samples
    if config.USING_S3 and config.S3_CACHE:
        report["cache_dir"] = str(config.CACHE_DIR)
    report["ok"] = True
    if verbose:
        print("OK            : source reachable and readable")
    return report
