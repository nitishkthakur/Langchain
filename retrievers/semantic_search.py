"""
Semantic Search Tool

This module exposes a `semantic_search` tool for LangChain agents. It runs a
natural language search for relevant code or documentation comments from the
user's current workspace using embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from pydantic import Field

# Configuration
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 5
MAX_FILE_SIZE = 100_000  # Skip files larger than 100KB
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks


def _get_embeddings() -> OllamaEmbeddings:
    """Get the Ollama embeddings model."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _collect_files(base: Path) -> list[Path]:
    """Collect all text files in the workspace."""
    files = []
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
        # Skip large files
        try:
            if file_path.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue
        files.append(file_path)
    return files


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


@tool
def semantic_search(
    query: Annotated[
        str,
        Field(
            description=(
                "The query to search the codebase for. Should contain all relevant "
                "context. Should ideally be text that might appear in the codebase, "
                "such as function names, variable names, or comments."
            )
        ),
    ],
    base_path: Annotated[
        str,
        Field(
            description=(
                "Directory root used as the workspace. "
                "Provide an absolute path or '.' to use the current workspace."
            )
        ),
    ] = ".",
    top_k: Annotated[
        int,
        Field(
            description="Number of top results to return.",
            ge=1,
            le=20,
        ),
    ] = DEFAULT_TOP_K,
) -> dict:
    """
    Run a natural language search for relevant code or documentation comments
    from the user's current workspace.

    Returns relevant code snippets from the user's current workspace if it is
    large, or the full contents of the workspace if it is small.
    """
    base = Path(base_path).expanduser().resolve()

    if not base.exists():
        return {"error": f"Base path not found: {base}"}
    if not base.is_dir():
        return {"error": f"Base path is not a directory: {base}"}

    # Collect files
    files = _collect_files(base)
    if not files:
        return {"error": "No files found in workspace", "base_path": str(base)}

    # Build chunks with metadata
    chunks_data: list[dict] = []
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            relative_path = str(file_path.relative_to(base))
            chunks = _chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    "file": relative_path,
                    "chunk_index": i,
                    "content": chunk,
                })
        except (OSError, UnicodeDecodeError):
            continue

    if not chunks_data:
        return {"error": "No readable content found in workspace", "base_path": str(base)}

    # Get embeddings
    try:
        embeddings = _get_embeddings()
        query_embedding = embeddings.embed_query(query)
        chunk_texts = [c["content"] for c in chunks_data]
        chunk_embeddings = embeddings.embed_documents(chunk_texts)
    except Exception as exc:
        return {"error": f"Failed to generate embeddings: {exc}"}

    # Calculate similarities and rank
    for i, chunk in enumerate(chunks_data):
        chunk["similarity"] = _cosine_similarity(query_embedding, chunk_embeddings[i])

    # Sort by similarity and get top results
    chunks_data.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = chunks_data[:top_k]

    # Format results
    results = []
    for chunk in top_results:
        results.append({
            "file": chunk["file"],
            "similarity": round(chunk["similarity"], 4),
            "snippet": chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"],
        })

    return {
        "base_path": str(base),
        "query": query,
        "total_chunks_searched": len(chunks_data),
        "result_count": len(results),
        "results": results,
    }
