"""
Grep Search Tool

This module exposes a `grep_search` tool for LangChain agents. It performs fast
text search in the workspace using exact strings or regex patterns, returning
structured results that are easy for an LLM to reason about.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

# Default limit for matches returned
DEFAULT_MAX_RESULTS = 100


@tool
def grep_search(
    query: Annotated[
        str,
        Field(
            description=(
                "The pattern to search for in files in the workspace. Use regex with "
                "alternation (e.g., 'word1|word2|word3') or character classes to find "
                "multiple potential words in a single search. Be sure to set the isRegexp "
                "property properly to declare whether it's a regex or plain text pattern. "
                "Is case-insensitive."
            )
        ),
    ],
    isRegexp: Annotated[
        bool,
        Field(
            description="Whether the pattern is a regex."
        ),
    ],
    includePattern: Annotated[
        str | None,
        Field(
            description=(
                "Search files matching this glob pattern. Will be applied to the relative "
                "path of files within the workspace. To search recursively inside a folder, "
                "use a proper glob pattern like \"src/folder/**\". Do not use | in includePattern."
            )
        ),
    ] = None,
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
    Do a fast text search in the workspace.

    Use this tool when you want to search with an exact string or regex. If you
    are not sure what words will appear in the workspace, prefer using regex
    patterns with alternation (|) or character classes to search for multiple
    potential words at once instead of making separate searches. For example,
    use 'function|method|procedure' to look for all of those words at once.
    Use includePattern to search within files matching a specific pattern, or
    in a specific file, using a relative path. Use this tool when you want to
    see an overview of a particular file, instead of using read_file many times
    to look for code within a file.
    """
    max_results = maxResults if maxResults is not None else DEFAULT_MAX_RESULTS

    if isRegexp:
        try:
            compiled = re.compile(query, re.IGNORECASE)
        except re.error as exc:
            return {"error": f"Invalid regular expression: {exc}"}
    else:
        # Escape special regex characters for plain text search
        compiled = re.compile(re.escape(query), re.IGNORECASE)

    base = Path(base_path).expanduser().resolve()

    if not base.exists():
        return {"error": f"Base path not found: {base}"}
    if not base.is_dir():
        return {"error": f"Base path is not a directory: {base}"}

    # Collect files to search
    files_to_search: list[Path] = []
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
        # Apply includePattern if provided
        if includePattern:
            if not fnmatch.fnmatch(str(relative), includePattern):
                continue
        files_to_search.append(file_path)

    matches: list[dict] = []
    truncated = False

    for file_path in files_to_search:
        if truncated:
            break
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, start=1):
                    for match in compiled.finditer(line):
                        relative_path = str(file_path.relative_to(base))
                        matches.append(
                            {
                                "file": relative_path,
                                "line": line_no,
                                "column_start": match.start() + 1,
                                "column_end": match.end(),
                                "match": match.group(),
                                "line_text": line.rstrip("\n"),
                            }
                        )
                        if len(matches) >= max_results:
                            truncated = True
                            break
                    if truncated:
                        break
        except (OSError, UnicodeDecodeError):
            # Skip files that can't be read
            continue

    return {
        "base_path": str(base),
        "query": query,
        "isRegexp": isRegexp,
        "includePattern": includePattern,
        "match_count": len(matches),
        "truncated": truncated,
        "matches": matches,
    }
