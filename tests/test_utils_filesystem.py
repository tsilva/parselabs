import os
import stat

import pytest

from parselabs.utils import make_path_writable


def test_make_path_writable_clears_macos_immutable_flag(tmp_path):
    if not hasattr(os, "chflags") or not hasattr(stat, "UF_IMMUTABLE"):
        pytest.skip("Filesystem flags are not available on this platform")

    path = tmp_path / "locked.txt"
    path.write_text("temporary", encoding="utf-8")
    os.chflags(path, stat.UF_IMMUTABLE)

    make_path_writable(path, 0o600)

    assert path.stat().st_flags & stat.UF_IMMUTABLE == 0
    path.unlink()
