"""Keep both Kotlin registry verification pins tied to the release version."""

import pytest
from check_kotlin_consumer_version import verify_coordinates

CURRENT = 'implementation("io.xberg.literllm:liter-llm-android:1.19.3")\n'
AAR = 'val aarCoord = "io.xberg.literllm:liter-llm-android:1.19.3"\n'


def test_current_coordinates_pass() -> None:
    """Both independent consumers must select the canonical release."""
    verify_coordinates(CURRENT + AAR, "1.19.3")


@pytest.mark.parametrize(
    "content",
    [
        CURRENT.replace("1.19.3", "1.16.0") + AAR,
        CURRENT + AAR.replace("1.19.3", "1.16.0"),
        CURRENT,
        "",
        CURRENT + AAR + AAR,
        CURRENT + CURRENT,
    ],
)
def test_stale_missing_or_duplicate_coordinate_fails(content: str) -> None:
    """No stale pin or zero-work result may certify the registry harness."""
    with pytest.raises(SystemExit, match="Kotlin consumer"):
        verify_coordinates(content, "1.19.3")


@pytest.mark.parametrize("content", ["// " + CURRENT + "// " + AAR, "/* " + CURRENT + AAR + " */"])
def test_commented_declarations_do_not_count(content: str) -> None:
    """Commented source cannot stand in for live registry checks."""
    with pytest.raises(SystemExit, match="Kotlin consumer"):
        verify_coordinates(content, "1.19.3")


def test_urls_and_comment_delimiters_inside_strings_are_preserved() -> None:
    """String data must not hide the live declarations that follow it."""
    verify_coordinates('val url = "https://example.test/*path*/"\n' + CURRENT + AAR, "1.19.3")


def test_nested_block_comments_do_not_expose_declarations() -> None:
    """Kotlin nested comments must remain comments after their inner close."""
    with pytest.raises(SystemExit, match="Kotlin consumer"):
        verify_coordinates("/* outer /* inner */ " + CURRENT + AAR + " */", "1.19.3")
