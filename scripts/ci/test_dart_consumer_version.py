"""Keep the published Dart consumer tied to the canonical release."""

import pytest
from sync_dart_consumer_version import synchronize

MANIFEST = "name: e2e_dart\nversion: 0.1.0\ndependencies:\n  liter_llm: ^1.16.0\n  ffi: ^2.2.0\n"


def test_sync_replaces_only_the_package_requirement() -> None:
    """A major release must update the consumer without changing its own version."""
    expected = MANIFEST.replace("^1.16.0", "2.0.0")
    assert synchronize(MANIFEST, "2.0.0") == expected
    assert synchronize(expected, "2.0.0") == expected


@pytest.mark.parametrize("content", ["", "  # liter_llm: ^1.16.0\n", MANIFEST + "  liter_llm: 1.18.4\n"])
def test_missing_or_duplicate_requirement_fails(content: str) -> None:
    """A zero-work or ambiguous edit cannot certify a release pin."""
    with pytest.raises(SystemExit, match="exactly one"):
        synchronize(content, "2.0.0")
