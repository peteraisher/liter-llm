"""Verify every Zig E2E source is wired into the test build."""

import re
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "e2e" / "zig"
    sources = {path.name for path in (root / "src").glob("*_test.zig")}
    build = re.sub(r"/\*.*?\*/|//[^\n]*", "", (root / "build.zig").read_text(), flags=re.DOTALL)
    targets = set(re.findall(r'b\.path\("src/([^"/]+_test\.zig)"\)', build))
    if not sources:
        raise SystemExit("No Zig E2E sources found")
    if targets != sources:
        raise SystemExit(
            f"Zig source/target mismatch: missing={sorted(sources - targets)}, stale={sorted(targets - sources)}"
        )
    for source in sorted(sources):
        category = source.removesuffix("_test.zig")
        if f"test_step.dependOn(&{category}_run.step);" not in build:
            raise SystemExit(f"Zig target is not connected to test: {source}")
    print(f"Zig census: {len(sources)} sources and connected targets")


if __name__ == "__main__":
    main()
