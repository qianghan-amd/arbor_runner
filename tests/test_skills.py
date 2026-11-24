"""The agent skill lives in skills/ (any agent tool reads it) and in .claude/skills/ (where Claude Code finds it).
Windows has no symlinks here, so the two are copies: this test keeps them identical."""
from arbor_ab import REPO_ROOT


def test_skill_copies_identical():
    src = REPO_ROOT / "skills" / "arbor-ci-run" / "SKILL.md"
    copy = REPO_ROOT / ".claude" / "skills" / "arbor-ci-run" / "SKILL.md"
    assert src.read_bytes() == copy.read_bytes(), f"copy {src} to {copy}"
    assert b"\r" not in src.read_bytes()
    assert src.read_text(encoding="utf-8").startswith("---\nname: arbor-ci-run\n")
