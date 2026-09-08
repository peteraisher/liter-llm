"""Prove the installed documentation rules inspect Markdown and MDX."""

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_case(path: Path, modules: Path, expected_status: int, marker: str = "") -> None:
    result = subprocess.run(
        [
            str(ROOT / "docs-site/node_modules/.bin/textlint"),
            "--config",
            str(ROOT / ".textlintrc.json"),
            "--rules-base-directory",
            str(modules),
            str(path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != expected_status or marker not in result.stdout + result.stderr:
        raise RuntimeError(
            f"Prose control {path.name} failed: expected exit {expected_status}, got {result.returncode}"
        )


def main() -> None:
    modules = ROOT / "docs-site/node_modules"
    cases = 0
    with tempfile.TemporaryDirectory(prefix="liter-prose-gate-") as directory:
        scratch = Path(directory)
        for extension in ("md", "mdx"):
            path = scratch / f"control.{extension}"
            path.write_text("The client sends a request.\n")
            check_case(path, modules, 0)
            cases += 1
            path.write_text("A zero\u200bwidth character.\n")
            check_case(path, modules, 1, "no-zero-width-spaces")
            cases += 1
        check_case(scratch / "control.md", scratch, 1, "No rules found")
        check_case(scratch / "absent.md", modules, 2)
        cases += 2
    if cases != 6:
        raise RuntimeError("Expected six real prose controls")
    print("Six real prose controls passed: Markdown, MDX, missing rules, and empty input")


if __name__ == "__main__":
    main()
