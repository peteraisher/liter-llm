"""Keep Apple deployment settings confined to Swift build tasks."""

import os
import subprocess
from pathlib import Path


def test_importing_swift_tasks_does_not_change_other_language_targets(tmp_path: Path) -> None:
    """A C or Ruby compiler must not inherit an iOS target from Swift tasks."""
    swift_tasks = Path(__file__).resolve().parents[2] / ".task/languages/swift.yml"
    taskfile = tmp_path / "Taskfile.yml"
    taskfile.write_text(
        f'version: "3"\nincludes:\n  swift: "{swift_tasks}"\ntasks:\n'
        '  probe:\n    cmds:\n      - printf "%s|%s" "$IPHONEOS_DEPLOYMENT_TARGET" "$MACOSX_DEPLOYMENT_TARGET"\n'
    )
    environment = os.environ.copy()
    environment.pop("IPHONEOS_DEPLOYMENT_TARGET", None)
    environment.pop("MACOSX_DEPLOYMENT_TARGET", None)
    result = subprocess.run(
        ["task", "--taskfile", str(taskfile), "probe"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "|"
