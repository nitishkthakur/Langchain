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
    instructions="""Workspace filesystem operations. All paths sandboxed to workspace root.

## Common Workflows:

### Create new file:
create_file(path="src/new.py", content="# code here")

### Edit existing file:
1. read_file(path="src/main.py") → see content
2. edit_file(path, old_string="exact text", new_string="replacement")

### Find and modify:
1. grep(pattern="TODO") → find locations
2. read_file → get context
3. edit_file → make change

### Explore structure:
list_path(path=".", recursive=True) → see all files

## Tool Selection:
- create_file: NEW files only (fails if exists)
- write_file: Create OR overwrite (set overwrite=True)
- edit_file: Surgical text replacement (read_file first!)
- read_file: View content before editing
- list_path: Directory listing
- grep: Search across files""",
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
def create_directory(
    path: Annotated[str, Field(description="Directory path. Relative ('src/utils') or absolute. Parent dirs created if parents=True.")],
    parents: Annotated[bool, Field(description="Create parent dirs if missing (like mkdir -p). True = create parents, False = fail if parent missing. Defaults to True.")] = True,
) -> dict:
    """Create a directory. Use before create_file for new folder structures. Safe: returns 'already_exists' if directory exists. Returns: path, status (created/already_exists)."""
    resolved = _resolve_path(path)
    
    if resolved.exists():
        if resolved.is_dir():
            return {
                "path": _relative(resolved),
                "status": "already_exists",
                "message": "Directory already exists."
            }
        else:
            return {"error": f"Path exists but is not a directory: {_relative(resolved)}"}
    
    try:
        resolved.mkdir(parents=parents, exist_ok=False)
    except FileNotFoundError:
        return {"error": f"Parent directory does not exist. Set parents=True to create parent directories."}
    except OSError as exc:
        return {"error": f"Failed to create directory: {exc}"}
    
    return {
        "path": _relative(resolved),
        "status": "created",
        "parents_created": parents,
    }


@mcp.tool()
def create_file(
    path: Annotated[str, Field(description="File path. Example: 'src/main.py', 'config/settings.json'. Parent dirs auto-created if create_parents=True.")],
    content: Annotated[str, Field(description="File content. Use '' for empty file. Include proper formatting/indentation.")] = "",
    create_parents: Annotated[bool, Field(description="Auto-create parent directories if missing. Defaults to True.")] = True,
    encoding: Annotated[str, Field(description="Text encoding. Usually 'utf-8'. Defaults to 'utf-8'.")] = "utf-8",
) -> dict:
    """Create NEW file with content. FAILS if file exists (safe for scaffolding). Use write_file(overwrite=True) to replace existing files. Returns: path, bytes_written."""
    resolved = _resolve_path(path)
    
    if resolved.exists():
        return {"error": f"File already exists: {_relative(resolved)}. Use write_file with overwrite=True to replace, or edit_file to modify."}
    
    if resolved.parent.exists() and not resolved.parent.is_dir():
        return {"error": f"Parent path exists but is not a directory: {_relative(resolved.parent)}"}
    
    try:
        if create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        elif not resolved.parent.exists():
            return {"error": f"Parent directory does not exist: {_relative(resolved.parent)}. Set create_parents=True or use create_directory first."}
        
        with resolved.open("w", encoding=encoding, errors="replace") as handle:
            handle.write(content)
    except OSError as exc:
        return {"error": f"Failed to create file: {exc}"}
    
    return {
        "path": _relative(resolved),
        "status": "created",
        "bytes_written": len(content.encode(encoding, errors="replace")),
        "has_content": len(content) > 0,
    }


@mcp.tool()
def edit_file(
    path: Annotated[str, Field(description="File to edit. Must exist. Use read_file first to see content.")],
    old_string: Annotated[str, Field(description="EXACT text to find. Must match perfectly including whitespace/newlines. Include 2-3 context lines for uniqueness. Copy from read_file output.")],
    new_string: Annotated[str, Field(description="Replacement text. Use '' to delete old_string. Preserve indentation and surrounding context.")],
    encoding: Annotated[str, Field(description="Text encoding. Defaults to 'utf-8'.")] = "utf-8",
) -> dict:
    """Replace EXACT text in file. Workflow: 1) read_file to see content, 2) copy exact text including whitespace, 3) edit_file with replacement. FAILS if: text not found, or matches multiple locations. For full rewrites use write_file(overwrite=True)."""
    resolved = _resolve_path(path)
    
    if not resolved.exists():
        return {"error": f"File not found: {_relative(resolved)}. Use create_file to create new files."}
    
    if not resolved.is_file():
        return {"error": f"Not a file: {_relative(resolved)}"}
    
    if not old_string:
        return {"error": "old_string cannot be empty. Provide the exact text to replace."}
    
    try:
        original_content = resolved.read_text(encoding=encoding, errors="replace")
    except OSError as exc:
        return {"error": f"Failed to read file: {exc}"}
    
    # Count occurrences
    occurrences = original_content.count(old_string)
    
    if occurrences == 0:
        # Provide helpful feedback
        return {
            "error": "old_string not found in file. Ensure exact match including whitespace, newlines, and indentation.",
            "path": _relative(resolved),
            "hint": "Use read_file to view the current file content and copy the exact text."
        }
    
    if occurrences > 1:
        return {
            "error": f"old_string found {occurrences} times. Include more context to make it unique (add surrounding lines).",
            "path": _relative(resolved),
            "occurrences": occurrences,
        }
    
    # Perform the replacement
    new_content = original_content.replace(old_string, new_string, 1)
    
    try:
        resolved.write_text(new_content, encoding=encoding, errors="replace")
    except OSError as exc:
        return {"error": f"Failed to write file: {exc}"}
    
    return {
        "path": _relative(resolved),
        "status": "edited",
        "chars_removed": len(old_string),
        "chars_added": len(new_string),
        "net_change": len(new_string) - len(old_string),
    }


@mcp.tool()
def list_path(
    path: Annotated[str, Field(description="Directory to list. '.' for workspace root, 'src' for src folder. Defaults to '.'.")] = ".",
    recursive: Annotated[bool, Field(description="True = include subdirectories (tree view). False = immediate children only. Defaults to False.")] = False,
    include_hidden: Annotated[bool, Field(description="True = include dotfiles (.git, .env). False = skip hidden. Defaults to False.")] = False,
    max_entries: Annotated[int, Field(description="Max entries to return. Lower = faster. Defaults to 2000.")] = 2000,
) -> dict:
    """List directory contents. Use to: explore workspace, find files before reading, understand project structure. Returns: array of {name, path, type, size, modified}."""
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
    path: Annotated[str, Field(description="File to read. Example: 'src/main.py', 'README.md'.")],
    start: Annotated[int, Field(description="Start line (1-based). 1 = beginning. Use to skip headers. Defaults to 1.")] = 1,
    num_lines: Annotated[Optional[int], Field(description="Lines to read. None = read all. 200 is good for context. 50 for snippets. Defaults to 200.")] = 200,
    encoding: Annotated[str, Field(description="Text encoding. Defaults to 'utf-8'.")] = "utf-8",
) -> dict:
    """Read file content. Use before edit_file to get exact text. Returns: content string, start/end line numbers, total_lines. For large files, use start+num_lines to read in chunks."""
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
    path: Annotated[str, Field(description="File path. Parent dirs created automatically.")],
    content: Annotated[str, Field(description="Full content to write. For partial updates use edit_file instead.")],
    append: Annotated[bool, Field(description="True = add to end of file. False = replace content. Defaults to False.")] = False,
    overwrite: Annotated[bool, Field(description="True = replace existing file. False = fail if exists. Required for updating files. Defaults to False.")] = False,
    encoding: Annotated[str, Field(description="Text encoding. Defaults to 'utf-8'.")] = "utf-8",
) -> dict:
    """Write content to file. Use for: new files, complete rewrites, appending. Set overwrite=True to replace existing. For surgical edits use edit_file. Returns: path, status, bytes_written."""
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
    pattern: Annotated[str, Field(description="Search text or regex. Examples: 'TODO', 'import.*pandas' (with use_regex=True), 'def process'.")],
    path: Annotated[str, Field(description="Search scope. '.' = workspace, 'src' = src folder, 'src/main.py' = single file. Defaults to '.'.")] = ".",
    use_regex: Annotated[bool, Field(description="True = regex pattern. False = literal text (safer). Defaults to False.")] = False,
    case_sensitive: Annotated[bool, Field(description="True = exact case. False = ignore case. Defaults to True.")] = True,
    include_hidden: Annotated[bool, Field(description="True = search dotfiles. False = skip hidden. Defaults to False.")] = False,
    max_results: Annotated[int, Field(description="Max matches. Lower = faster. Defaults to 200.")] = 200,
    max_file_size_kb: Annotated[int, Field(description="Skip files larger than this (KB). Defaults to 1024 (1MB).")] = 1024,
    encoding: Annotated[str, Field(description="File encoding. Defaults to 'utf-8'.")] = "utf-8",
) -> dict:
    """Search files for text/pattern. Returns: file paths, line numbers, matching text. Use for: finding code, locating definitions, searching TODOs. Like grep/ripgrep."""
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

### Directory & File Creation
- **create_directory**: Create folders. Use before create_file for new project structures.
- **create_file**: Create new files with optional content. Fails if file exists (safe for scaffolding).

### File Operations
- **read_file**: Read portions of a file. Use before edit_file to see exact content.
- **write_file**: Write/overwrite/append content. Use for complete file rewrites.
- **edit_file**: Replace specific text in a file. Use for targeted modifications.

### Exploration & Search
- **list_path**: List directory contents. Use to explore structure before operations.
- **grep**: Search for text/patterns across files.

## Common Workflows

### Create a new file in a new folder:
1. `create_directory(path="src/utils")` - Create the folder
2. `create_file(path="src/utils/helpers.py", content="# Helper functions")` - Create file with content

### Edit an existing file:
1. `read_file(path="src/main.py")` - View current content
2. `edit_file(path="src/main.py", old_string="old code", new_string="new code")` - Make changes

### Scaffold a project structure:
1. `create_directory(path="myproject/src")`
2. `create_directory(path="myproject/tests")`
3. `create_file(path="myproject/src/__init__.py")`
4. `create_file(path="myproject/README.md", content="# My Project")`

### Explore and modify:
1. `list_path(path=".", recursive=True)` - See all files
2. `grep(pattern="TODO", path="src")` - Find TODOs
3. `read_file(path="src/module.py")` - Read the file
4. `edit_file(...)` - Make targeted edits

## Usage Notes
- All paths are resolved relative to the workspace root to prevent escapes.
- `create_file` fails if file exists; use `write_file(overwrite=True)` to replace.
- `edit_file` requires exact text match; use `read_file` first to get exact content.
- Use `include_hidden=True` to surface dotfiles in list_path and grep.
"""


if __name__ == "__main__":
    mcp.run()
