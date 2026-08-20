"""Operational CLI for the dataset source — `python -m dataset <cmd>`.

    check          resolve the source, list the subtree, read a few CSVs end-to-end
    warm           download the whole router-interfaces subtree into the local cache
    cache-info     where the cache lives and how big it is
    cache-clear    delete the cached objects and listings

``check`` is the first thing to run on a new box: it separates a credentials or
endpoint problem from a pipeline problem.
"""
from __future__ import annotations

import argparse
import sys

from . import config, loaders, storage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dataset", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command",
                    choices=["check", "warm", "cache-info", "cache-clear"])
    ap.add_argument("--sample", type=int, default=3,
                    help="CSVs to read end-to-end during check (default 3)")
    args = ap.parse_args(argv)

    if args.command == "check":
        return 0 if storage.check(sample=args.sample)["ok"] else 1

    if args.command == "warm":
        print(f"source: {config.describe_source()}")
        if not (config.USING_S3 and config.S3_CACHE):
            print("nothing to warm (local source, or GR_S3_CACHE=0)")
            return 0
        n = loaders.prefetch_interfaces(verbose=True)
        print(f"downloaded {n} objects; cache now "
              f"{storage.cache_size_bytes() / 1e9:.1f} GB")
        return 0

    if args.command == "cache-info":
        print(f"source    : {config.describe_source()}")
        print(f"cache dir : {config.CACHE_DIR}")
        print(f"enabled   : {config.S3_CACHE}   validate: {config.S3_VALIDATE}")
        if config.CACHE_DIR.exists():
            print(f"size      : {storage.cache_size_bytes() / 1e9:.2f} GB")
        else:
            print("size      : (not created yet)")
        return 0

    storage.clear_cache()
    print(f"cleared {config.CACHE_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        sys.exit(2)
