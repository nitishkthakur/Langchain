"""
Line Retriever Tool

This module exposes a `line_retriever` tool for LangChain agents. It retrieves
specific lines or ranges of lines from a single text file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

# Hard limit to keep responses concise and safe for LLM consumption
MAX_LINES = 500


@tool
def line_retriever(
    file_name: Annotated[
        str,
        Field(
            description=(
                "File name (relative to base_path) to inspect. "
                "Example: 'src/utils.py', 'README.md'. "
            )
        ),
    ],
    start_line: Annotated[
        int,
        Field(
            description="The starting line number (1-based, inclusive).",
            ge=1,
        ),
    ],
    end_line: Annotated[
        int,
        Field(
            description="The ending line number (1-based, inclusive).",
            ge=1,
        ),
    ],
    base_path: Annotated[
        str,
        Field(
            description=(
                "Directory root used to resolve file_name. "
                "Keeps searches scoped (e.g., the repository root). "
                "Provide an absolute path or '.' to use the current workspace."
            )
        ),
    ] = ".",
) -> dict:
    """
    Retrieve a range of lines from a single file.

    LLM usage guidance:
    - Call this when you need to read specific lines from a file, for example
      after a search tool gave you line numbers or to read a chunk of code.
    - Line numbers are 1-based.
    - If the requested range exceeds the file length, it will be truncated to the end of the file.
    """
    base = Path(base_path).expanduser().resolve()
    target = Path(file_name)
    if not target.is_absolute():
        target = (base / target).resolve()

    try:
        target.relative_to(base)
    except ValueError:
        return {
            "error": (
                f"File is outside base_path. Adjust base_path to include the file."
            ),
            "base_path": str(base),
            "requested_file": str(target),
        }

    if not target.exists():
        return {"error": f"File not found: {target}", "base_path": str(base)}
    if not target.is_file():
        return {"error": f"Not a file: {target}", "base_path": str(base)}

    lines: list[str] = []
    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return {"error": f"Failed to read file: {exc}", "path": str(target)}

    total_lines = len(lines)
    
    # Ensure start_line is at least 1
    safe_start = max(1, start_line)
    # Ensure end_line is not beyond total_lines
    safe_end = min(total_lines, end_line)

    if safe_start > safe_end:
         return {
            "error": f"Invalid range: start_line ({start_line}) > end_line ({end_line}) or start_line > total_lines ({total_lines}).",
            "total_lines": total_lines
        }
    
    if (safe_end - safe_start + 1) > MAX_LINES:
        return {
            "error": f"Requested line count ({safe_end - safe_start + 1}) exceeds maximum limit of {MAX_LINES}.",
            "max_lines": MAX_LINES
        }

    retrieved_lines = lines[safe_start - 1 : safe_end]
    content = "".join(retrieved_lines)

    relative_path = str(target.relative_to(base))
    return {
        "file": relative_path,
        "base_path": str(base),
        "start_line": safe_start,
        "end_line": safe_end,
        "total_lines": total_lines,
        "content": content,
    }
