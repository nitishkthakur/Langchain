"""
File Search Tool

This module exposes a `file_search` tool for LangChain agents. It searches for
files in the workspace by glob pattern and returns the paths of matching files.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

# Default limit for results returned
DEFAULT_MAX_RESULTS = 100


@tool
def file_search(
    query: Annotated[
        str,
        Field(
            description="Search for files with names or paths matching this glob pattern."
        ),
    ],
    maxResults: Annotated[
        int | None,
        Field(
            description=(
                "The maximum number of results to return. Do not use this unless necessary, "
                "it can slow things down. By default, only some matches are returned. If you "
                "use this and don't see what you're looking for, you can try again with a "
                "more specific query or a larger maxResults."
            ),
            ge=1,
        ),
    ] = None,
    base_path: Annotated[
        str,
        Field(
            description=(
                "Directory root used as the workspace. "
                "Provide an absolute path or '.' to use the current workspace."
            )
        ),
    ] = ".",
) -> dict:
    """
    Search for files in the workspace by glob pattern.

    This only returns the paths of matching files. Use this tool when you know
    the exact filename pattern of the files you're searching for. Glob patterns
    match from the root of the workspace folder. Examples:
    - **/*.{js,ts} to match all js/ts files in the workspace.
    - src/** to match all files under the top-level src folder.
    - **/foo/**/*.js to match all js files under any foo folder in the workspace.
    """
    max_results = maxResults if maxResults is not None else DEFAULT_MAX_RESULTS

    base = Path(base_path).expanduser().resolve()

    if not base.exists():
        return {"error": f"Base path not found: {base}"}
    if not base.is_dir():
        return {"error": f"Base path is not a directory: {base}"}

    # Collect all files in the workspace
    matching_files: list[str] = []
    truncated = False

    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue
        # Skip hidden files and common non-text directories
        relative = file_path.relative_to(base)
        parts = relative.parts
        if any(part.startswith(".") for part in parts):
            continue
        if any(part in ("node_modules", "__pycache__", ".git", "venv", ".venv") for part in parts):
            continue

        # Match against the glob pattern
        relative_str = str(relative)
        if fnmatch.fnmatch(relative_str, query):
            matching_files.append(relative_str)
            if len(matching_files) >= max_results:
                truncated = True
                break

    return {
        "base_path": str(base),
        "query": query,
        "file_count": len(matching_files),
        "truncated": truncated,
        "files": matching_files,
    }
