"""Small deterministic helpers for software-engineering training data."""
from __future__ import annotations

import re

_HUNK = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<tail>.*)$"
)
_INDEX = re.compile(r"^index (?P<old>[0-9A-Fa-f]+)\.\.(?P<new>[0-9A-Fa-f]+)(?P<tail>.*)$")


def _range(start: str, count: str | None) -> str:
    return start if count is None else f"{start},{count}"




def _git_path(path: str, prefix: str) -> str:
    if path == "/dev/null":
        return path
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return f"{prefix}/{path}"


def _header_path(value: str, prefix: str) -> str:
    if value == "/dev/null":
        return value
    if value.startswith(("a/", "b/")):
        return prefix + "/" + value[2:]
    return value


def invert_unified_diff(patch: str) -> str:
    """Return the inverse of a text unified diff.

    SWE-smith stores the patch that *introduces* the synthetic bug. Its gold
    repair is therefore the reverse patch. Binary patches are rejected because
    they cannot be safely reconstructed from a textual inverse.
    """
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("patch must be non-empty text")
    if "GIT binary patch" in patch or any(
        line.startswith("Binary files ") for line in patch.splitlines()
    ):
        raise ValueError("binary diffs are not supported")

    lines = patch.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("diff --git "):
            parts = line.split(" ", 3)
            if len(parts) == 4:
                left, right = parts[2], parts[3]
                out.append(f"diff --git {_git_path(right, 'a')} {_git_path(left, 'b')}")
            else:
                out.append(line)
            i += 1
            continue

        match = _INDEX.match(line)
        if match:
            out.append(
                f"index {match.group('new')}..{match.group('old')}{match.group('tail')}"
            )
            i += 1
            continue

        if line.startswith("new file mode "):
            out.append("deleted file mode " + line[len("new file mode "):])
            i += 1
            continue
        if line.startswith("deleted file mode "):
            out.append("new file mode " + line[len("deleted file mode "):])
            i += 1
            continue

        if line.startswith("old mode ") and i + 1 < len(lines) and lines[i + 1].startswith("new mode "):
            out.append("old mode " + lines[i + 1][len("new mode "):])
            out.append("new mode " + line[len("old mode "):])
            i += 2
            continue

        if line.startswith("rename from ") and i + 1 < len(lines) and lines[i + 1].startswith("rename to "):
            out.append("rename from " + lines[i + 1][len("rename to "):])
            out.append("rename to " + line[len("rename from "):])
            i += 2
            continue

        if line.startswith("copy from ") and i + 1 < len(lines) and lines[i + 1].startswith("copy to "):
            out.append("copy from " + lines[i + 1][len("copy to "):])
            out.append("copy to " + line[len("copy from "):])
            i += 2
            continue

        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            old_value = line[len("--- "):]
            new_value = lines[i + 1][len("+++ "):]
            out.append("--- " + _header_path(new_value, "a"))
            out.append("+++ " + _header_path(old_value, "b"))
            i += 2
            continue

        match = _HUNK.match(line)
        if match:
            out.append(
                "@@ -"
                + _range(match.group("new_start"), match.group("new_count"))
                + " +"
                + _range(match.group("old_start"), match.group("old_count"))
                + " @@"
                + match.group("tail")
            )
            i += 1
            continue

        if line.startswith("+") and not line.startswith("+++"):
            out.append("-" + line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            out.append("+" + line[1:])
        else:
            out.append(line)
        i += 1

    inverted = "\n".join(out)
    if patch.endswith("\n"):
        inverted += "\n"
    return inverted
