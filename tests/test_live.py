"""Opt-in checks against the real bucket (needs credentials and network):

    GR_TEST_LIVE=1 pytest -m live
"""
from __future__ import annotations

import os
import subprocess

import pytest

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.environ.get("GR_TEST_LIVE") != "1",
                                 reason="set GR_TEST_LIVE=1 to talk to the S3 bucket")]


def test_bucket_is_readable():
    p = subprocess.run(["gr-data", "check", "--sample", "2"], capture_output=True,
                       text=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
