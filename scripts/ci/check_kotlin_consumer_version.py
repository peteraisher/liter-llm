"""Check Kotlin's registry dependency and AAR verification against the release version."""

import re
from pathlib import Path

import tomllib

COORDINATE = "io.xberg.literllm:liter-llm-android"
EXPECTED_COORDINATES = 2


def without_comments(content: str) -> str:
    """Remove nested Kotlin comments without interpreting delimiters inside strings."""
    result = []
    index = 0
    depth = 0
    while index < len(content):
        if content.startswith("/*", index):
            depth += 1
            result.append(" ")
            index += 2
        elif depth and content.startswith("*/", index):
            depth -= 1
            index += 2
        elif depth:
            index += 1
        elif content.startswith("//", index):
            end = content.find("\n", index)
            index = len(content) if end < 0 else end
        elif content[index] == '"':
            match = re.match(r'"""[\s\S]*?"""|"(?:\\.|[^"\\])*"', content[index:])
            if match is None:
                raise SystemExit("Kotlin consumer has an unterminated string")
            result.append(match.group())
            index += match.end()
        else:
            result.append(content[index])
            index += 1
    if depth:
        raise SystemExit("Kotlin consumer has an unterminated block comment")
    return "".join(result)


def verify_coordinates(content: str, version: str) -> None:
    """Reject stale, missing, or duplicate release coordinates."""
    content = without_comments(content)
    coordinate = re.escape(COORDINATE)
    versions = re.findall(rf'{coordinate}:([^"\s]+)', content)
    if versions != [version] * EXPECTED_COORDINATES:
        raise SystemExit(f"Kotlin consumer must declare exactly two {COORDINATE}:{version} pins; found {versions}")
    for declaration in (r"implementation\(", r"val\s+aarCoord\s*=\s*"):
        matches = re.findall(rf'{declaration}"{coordinate}:([^"\s]+)"', content)
        if matches != [version]:
            raise SystemExit(f"Kotlin consumer declaration {declaration} must select {version}; found {matches}")


def main() -> None:
    """Verify the source pins without resolving an unpublished Maven package."""
    root = Path(__file__).resolve().parents[2]
    manifest = tomllib.loads((root / "Cargo.toml").read_text())
    version = manifest["workspace"]["package"]["version"]
    if not isinstance(version, str) or not version:
        raise SystemExit("Kotlin consumer cannot be checked without a canonical Cargo version")
    path = root / "test_apps/kotlin_android/build.gradle.kts"
    verify_coordinates(path.read_text(), version)
    print(f"Kotlin consumer: {EXPECTED_COORDINATES} coordinates match {version}; registry resolution is separate")


if __name__ == "__main__":
    main()
