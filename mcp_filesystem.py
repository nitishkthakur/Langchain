"""
MCP Filesystem Server

This server exposes a set of filesystem tools inspired by the GitHub Copilot
Agent's local utilities. It is intended for general-purpose workspace
introspection and editing in environments that support the MCP protocol.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Iterable, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

# Root all filesystem operations to this directory to avoid accidental access
# outside the workspace. Set MCP_FS_ROOT to override the default.
ROOT = Path(os.environ.get("MCP_FS_ROOT", Path.cwd())).resolve()

mcp = FastMCP(
    name="filesystem-server",
    instructions="""You expose safe, workspace-scoped filesystem utilities.

Use these tools to explore, read, search, and write files. Keep paths relative
to the configured root when possible and avoid unnecessary writes. When
returning results, keep outputs concise and mention relative paths.""",
)


def _resolve_path(path: str) -> Path:
    """Resolve a path against the workspace root and guard against escapes."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"Path {path!r} is outside the workspace root {ROOT}")
    return resolved


def _relative(path: Path) -> str:
    """Return a workspace-relative string for a resolved path."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _is_hidden(path: Path) -> bool:
    """Determine whether a path should be treated as hidden."""
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        return True
    return any(part.startswith(".") for part in parts if part)


def _describe_entry(path: Path) -> dict:
    """Summarize a filesystem entry for listing tools."""
    try:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
        size = stat.st_size
    except OSError:
        modified = None
        size = None

    if path.is_symlink():
        entry_type = "symlink"
    elif path.is_dir():
        entry_type = "directory"
    elif path.is_file():
        entry_type = "file"
    else:
        entry_type = "other"

    return {
        "name": path.name,
        "path": _relative(path),
        "type": entry_type,
        "size_bytes": size,
        "modified": modified,
    }


def _iter_files(base: Path, include_hidden: bool) -> Iterable[Path]:
    """Yield files under base, respecting hidden rules."""
    if base.is_file():
        yield base
        return

    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if not include_hidden and _is_hidden(path):
            continue
        yield path


@mcp.tool()
def list_path(
    path: Annotated[str, Field(description="Directory or file path to list. Absolute ('/home/user/project') or relative to workspace root ('./src'). Defaults to '.'")] = ".",
    recursive: Annotated[bool, Field(description="If True, descends into subdirectories. If False, lists only immediate children. Defaults to False.")] = False,
    include_hidden: Annotated[bool, Field(description="If True, includes dotfiles (e.g., '.gitignore'). If False, skips hidden entries. Defaults to False.")] = False,
    max_entries: Annotated[int, Field(description="Maximum entries to return. Defaults to 2000. Use lower values for faster results.", ge=1, le=10000)] = 2000,
) -> dict:
    """List files and directories at specified path. Use this for exploring workspace structure or locating files before reading them."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        return {"error": f"Path not found: {_relative(resolved)}"}

    entries = []

    if resolved.is_file():
        entries.append(_describe_entry(resolved))
    else:
        iterator = resolved.rglob("*") if recursive else resolved.iterdir()
        for item in iterator:
            if not include_hidden and _is_hidden(item):
                continue
            entries.append(_describe_entry(item))
            if len(entries) >= max_entries:
                break

    return {
        "root": str(ROOT),
        "target": _relative(resolved),
        "entry_count": len(entries),
        "entries": entries,
        "truncated": len(entries) >= max_entries,
    }


@mcp.tool()
def read_file(
    path: Annotated[str, Field(description="File path to read. Absolute ('/home/user/file.txt') or relative to workspace root ('./src/main.py').")],
    start: Annotated[int, Field(description="1-based line number to start reading from. Example: 1 for beginning, 50 to skip first 49 lines. Defaults to 1.", ge=1)] = 1,
    num_lines: Annotated[Optional[int], Field(description="Number of lines to read. If None, reads to end of file. Defaults to 200.")] = 200,
    encoding: Annotated[str, Field(description="Text encoding. Examples: 'utf-8', 'latin-1', 'cp1252'. Defaults to 'utf-8'.")] = "utf-8",
) -> dict:
    """Read lines from a text file. Use this for examining portions of large files without loading entire content into memory."""
    resolved = _resolve_path(path)
    if not resolved.is_file():
        return {"error": f"Not a file: {_relative(resolved)}"}

    start_line = max(1, start)
    max_lines = None if num_lines is None else max(0, num_lines)

    collected = []
    total_lines = 0
    end_line = start_line - 1

    try:
        with resolved.open("r", encoding=encoding, errors="replace") as handle:
            for idx, line in enumerate(handle, start=1):
                total_lines = idx
                if idx < start_line:
                    continue
                if max_lines is not None and len(collected) >= max_lines:
                    break
                collected.append(line)
                end_line = idx
    except OSError as exc:
        return {"error": f"Failed to read file: {exc}"}

    return {
        "path": _relative(resolved),
        "start": start_line,
        "end": end_line,
        "total_lines": total_lines,
        "content": "".join(collected),
    }


@mcp.tool()
def write_file(
    path: Annotated[str, Field(description="File path to write. Absolute or relative to workspace root. Parent directories are created automatically.")],
    content: Annotated[str, Field(description="Full text content to write to the file.")],
    append: Annotated[bool, Field(description="If True, adds content to end of existing file. If False, replaces content. Defaults to False.")] = False,
    overwrite: Annotated[bool, Field(description="If True, replaces existing file. If False, raises error if file exists. Defaults to False for safety.")] = False,
    encoding: Annotated[str, Field(description="Text encoding. Examples: 'utf-8', 'latin-1'. Defaults to 'utf-8'.")] = "utf-8",
) -> dict:
    """Write text content to a file. Use this for creating new files or modifying existing ones. Set overwrite=True or append=True explicitly for existing files."""
    resolved = _resolve_path(path)
    if append and overwrite:
        return {"error": "Choose either append or overwrite, not both."}

    existed_before = resolved.exists()

    if existed_before and resolved.is_dir():
        return {"error": f"Cannot write to directory: {_relative(resolved)}"}

    if existed_before and not append and not overwrite:
        return {
            "error": f"File exists: {_relative(resolved)}. Set overwrite=True or append=True."
        }

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with resolved.open(mode, encoding=encoding, errors="replace") as handle:
            handle.write(content)
    except OSError as exc:
        return {"error": f"Failed to write file: {exc}"}

    return {
        "path": _relative(resolved),
        "status": "appended" if append else "overwritten" if existed_before else "created",
        "bytes_written": len(content.encode(encoding, errors="replace")),
    }


@mcp.tool()
def grep(
    pattern: Annotated[str, Field(description="Search term or regex pattern. Examples: 'TODO', 'def.*calculate' (with use_regex=True).")],
    path: Annotated[str, Field(description="File or directory to search. Absolute or relative to workspace root. Defaults to '.' (entire workspace).")] = ".",
    use_regex: Annotated[bool, Field(description="If True, treats pattern as regex. If False, treats as literal text. Defaults to False for safety.")] = False,
    case_sensitive: Annotated[bool, Field(description="If True, matches exact case. If False, case-insensitive search. Defaults to True.")] = True,
    include_hidden: Annotated[bool, Field(description="If True, searches dotfiles. If False, skips hidden files. Defaults to False.")] = False,
    max_results: Annotated[int, Field(description="Maximum matches to return. Defaults to 200. Use lower values for faster results.", ge=1, le=1000)] = 200,
    max_file_size_kb: Annotated[int, Field(description="Skip files larger than this size in KB. Defaults to 1024 (1MB).", ge=1)] = 1024,
    encoding: Annotated[str, Field(description="Text encoding for reading files. Defaults to 'utf-8'.")] = "utf-8",
) -> dict:
    """Search files for text or regex patterns. Use this for finding code, comments, or specific strings across multiple files. Returns file paths, line numbers, and matching text."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        return {"error": f"Path not found: {_relative(resolved)}"}

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(pattern if use_regex else re.escape(pattern), flags=flags)
    except re.error as exc:
        return {"error": f"Invalid regular expression: {exc}"}

    matches = []
    skipped_large = []
    truncated = False

    for file_path in _iter_files(resolved, include_hidden=include_hidden):
        try:
            size_kb = file_path.stat().st_size / 1024
            if size_kb > max_file_size_kb:
                skipped_large.append(_relative(file_path))
                continue
        except OSError:
            continue

        try:
            with file_path.open("r", encoding=encoding, errors="replace") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if compiled.search(line):
                        matches.append(
                            {
                                "path": _relative(file_path),
                                "line": line_no,
                                "text": line.rstrip("\n"),
                            }
                        )
                        if len(matches) >= max_results:
                            truncated = True
                            break
        except OSError:
            continue

        if truncated:
            break

    return {
        "pattern": pattern,
        "path": _relative(resolved),
        "matches": matches,
        "match_count": len(matches),
        "truncated": truncated,
        "skipped_large_files": skipped_large,
    }


@mcp.resource("info://server")
def get_server_info() -> str:
    """Summarize the filesystem tools and usage tips."""
    return f"""# Filesystem Server

Workspace root: `{ROOT}`

## Available Tools
- list_path: List files or folders, optionally recursively.
- read_file: Read a slice of a file with 1-based line offsets.
- write_file: Create, overwrite, or append text to files.
- grep: Search for patterns across files with regex support.

## Usage Notes
- All paths are resolved relative to the workspace root to prevent escapes.
- Use include_hidden=true to surface dotfiles.
- Provide overwrite=true or append=true explicitly to modify existing files.
"""


if __name__ == "__main__":
    mcp.run()
