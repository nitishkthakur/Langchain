"""
List Directory Tool

This module exposes a `list_dir` tool for LangChain agents. It lists the
contents of a directory, indicating whether each item is a file or folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
def list_dir(
    path: Annotated[
        str,
        Field(
            description="The absolute path to the directory to list."
        ),
    ],
) -> dict:
    """
    List the contents of a directory.

    Result will have the name of the child. If the name ends in /, it's a
    folder, otherwise a file.
    """
    target = Path(path).expanduser().resolve()

    if not target.exists():
        return {"error": f"Directory not found: {target}"}
    if not target.is_dir():
        return {"error": f"Not a directory: {target}"}

    try:
        children: list[str] = []
        for item in sorted(target.iterdir()):
            # Skip hidden files/folders
            if item.name.startswith("."):
                continue
            if item.is_dir():
                children.append(f"{item.name}/")
            else:
                children.append(item.name)
    except OSError as exc:
        return {"error": f"Failed to list directory: {exc}", "path": str(target)}

    return {
        "path": str(target),
        "count": len(children),
        "children": children,
    }
