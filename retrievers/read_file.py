"""
Read File Tool

This module exposes a `read_file` tool for LangChain agents. It reads the
contents of a file within a specified line range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
def read_file(
    filePath: Annotated[
        str,
        Field(
            description="The absolute path of the file to read."
        ),
    ],
    startLine: Annotated[
        int,
        Field(
            description="The line number to start reading from, 1-based.",
            ge=1,
        ),
    ],
    endLine: Annotated[
        int,
        Field(
            description="The inclusive line number to end reading at, 1-based.",
            ge=1,
        ),
    ],
) -> dict:
    """
    Read the contents of a file.

    You must specify the line range you're interested in. Line numbers are
    1-indexed. If the file contents returned are insufficient for your task,
    you may call this tool again to retrieve more content. Prefer reading
    larger ranges over doing many small reads.
    """
    target = Path(filePath).expanduser().resolve()

    if not target.exists():
        return {"error": f"File not found: {target}"}
    if not target.is_file():
        return {"error": f"Not a file: {target}"}

    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return {"error": f"Failed to read file: {exc}", "path": str(target)}

    total_lines = len(lines)

    if startLine > endLine:
        return {
            "error": f"Invalid range: startLine ({startLine}) > endLine ({endLine}).",
            "total_lines": total_lines,
        }

    if startLine > total_lines:
        return {
            "error": f"startLine ({startLine}) exceeds total lines ({total_lines}).",
            "total_lines": total_lines,
        }

    # Clamp endLine to total_lines
    safe_end = min(endLine, total_lines)

    retrieved_lines = lines[startLine - 1 : safe_end]
    content = "".join(retrieved_lines)

    return {
        "filePath": str(target),
        "startLine": startLine,
        "endLine": safe_end,
        "total_lines": total_lines,
        "content": content,
    }
