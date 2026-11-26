"""scripts/arbor_ab/github_relay.py: static checks on the remote script (it runs on the relay host over ssh, so a
live test needs a real relay; these catch a regression of the bug below without one)."""
import re

from arbor_ab import github_relay as gh


def test_every_zipfile_open_is_closed_before_the_tempdir_is_removed():
    # 2026-11-26 (CI run 45587499093): a bare `z = zipfile.ZipFile(zp)` left the zip's file handle open; on the relay
    # host's NFS home, TemporaryDirectory's rmtree then hit "OSError: [Errno 39] Directory not empty" (NFS silly-
    # renames a file still open at unlink), 3/3 reproducible for that artifact. Every call opened inside a
    # `with tempfile.TemporaryDirectory()` must close its ZipFile first (`with zipfile.ZipFile(...) as z:`).
    opens = re.findall(r"zipfile\.ZipFile\([^)]*\)", gh.REMOTE)
    assert opens, "no zipfile.ZipFile(...) calls found: did the remote script change shape?"
    for m in re.finditer(r"^(\s*)(\S.*zipfile\.ZipFile\([^)]*\).*)$", gh.REMOTE, re.M):
        line = m.group(2)
        assert line.strip().startswith("with ") and " as " in line, (
            f"zipfile.ZipFile opened without `with ... as z:` (never closed): {line!r}")
