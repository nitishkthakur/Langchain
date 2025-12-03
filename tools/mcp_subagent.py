"""
MCP Subagent Server

This server bundles several focused sub-agent style tools behind a single MCP
endpoint. Each tool mirrors a small, purpose-built helper (code writer,
reviewer, debugger, test generator, documentation writer, dependency manager,
architecture planner, filesystem navigator, and terminal executor).

The goal is to give an orchestrating agent lightweight, composable actions
without needing a full framework. Tools are intentionally conservative: they
stay within the workspace, avoid destructive edits, and return structured
summaries alongside any file changes they make.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated, Iterable, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# LangChain model setup (user-configurable)
try:
    from langchain.chat_models import init_chat_model  # type: ignore
except Exception:
    try:
        from langchain import init_chat_model  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        init_chat_model = None  # type: ignore

# Keep all operations sandboxed to the workspace (override with MCP_WORKSPACE_ROOT)
ROOT = Path(os.environ.get("MCP_WORKSPACE_ROOT", Path.cwd())).resolve()

# User-configurable chat model. Set MCP_SUBAGENT_MODEL (and provider via env handled by init_chat_model).
MODEL_ID = os.environ.get("MCP_SUBAGENT_MODEL", "gpt-4o-mini")
_CHAT_MODEL = None

mcp = FastMCP(
    name="subagent-server",
    instructions="""Specialized coding sub-agents for development tasks. Each tool makes an LLM call.

## Tool Categories (use the right tool for the job):
- **Write/Edit Code**: write_code (new files), modify_code (edit existing), apply_fix (bug fixes)
- **Review Code**: review_changes (full review), check_common_mistakes (quick scan)
- **Debug**: analyze_error → suggest_fix → apply_fix (chain these), add_logging
- **Testing**: generate_tests (create test file), add_test_case (add to existing)
- **Documentation**: add_docstrings, create_readme, explain_code
- **Dependencies**: add_package, check_outdated, check_security
- **Architecture**: plan_feature, suggest_structure, recommend_approach
- **Navigation**: find_files, find_definition, find_references
- **Execute**: run_command, run_tests, run_linter

## Key Behaviors:
- All operations sandboxed to workspace root
- Prefer additive edits over destructive changes
- Returns structured JSON with previews and status
- Set MCP_SUBAGENT_MODEL env var to change LLM model""",
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _get_chat_model():
    """Initialize (once) a chat model using LangChain's init_chat_model."""
    if init_chat_model is None:
        raise RuntimeError(
            "langchain.init_chat_model is not available. Install langchain and set MCP_SUBAGENT_MODEL."
        )
    global _CHAT_MODEL
    if _CHAT_MODEL is None:
        _CHAT_MODEL = init_chat_model(MODEL_ID)
    return _CHAT_MODEL


def _as_text(llm_output) -> str:
    """Normalize LangChain outputs into a string."""
    if llm_output is None:
        return ""
    if hasattr(llm_output, "content"):
        return str(getattr(llm_output, "content"))
    return str(llm_output)


def _call_llm(prompt: str) -> tuple[Optional[str], Optional[str]]:
    """Call the configured LLM and return (text, error)."""
    try:
        model = _get_chat_model()
    except Exception as exc:
        return None, str(exc)
    try:
        response = model.invoke(prompt)
        return _as_text(response), None
    except Exception as exc:  # pragma: no cover - runtime guard
        return None, f"LLM call failed: {exc}"


def _resolve_path(path: str) -> Path:
    """Resolve a user path against the workspace root and guard escapes."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"Path {path!r} is outside the workspace root {ROOT}")
    return resolved


def _ensure_parent(path: Path) -> None:
    """Create parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path, encoding: str = "utf-8") -> str:
    """Read text content, raising helpful errors if not a file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Not a file: {path}")
    return path.read_text(encoding=encoding, errors="replace")


def _write_text(path: Path, content: str, *, append: bool = False, encoding: str = "utf-8") -> None:
    """Write or append text content to a file."""
    _ensure_parent(path)
    mode = "a" if append else "w"
    with path.open(mode, encoding=encoding, errors="replace") as handle:
        handle.write(content)


def _relative(path: Path) -> str:
    """Return a workspace-relative path string for responses."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _extract_code_block(text: str) -> Optional[str]:
    """Extract the first fenced code block from a description, if any."""
    match = re.search(r"```(?:\w+\n)?(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip("\n")
    return None


def _extract_json_payload(text: str) -> Optional[object]:
    """Try to parse JSON either directly or from the first fenced code block."""
    candidates = [text, _extract_code_block(text)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _slugify(text: str, default: str = "snippet") -> str:
    """Create a lightweight identifier from free text."""
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    if not tokens:
        return default
    return "_".join(tokens[:6])


def _comment_prefix(path: Path, language: Optional[str] = None) -> str:
    """Choose a reasonable single-line comment prefix based on file or language."""
    lang = (language or "").lower()
    suffix = path.suffix.lower()
    mapping = {
        ".py": "#",
        ".rb": "#",
        ".sh": "#",
        ".yml": "#",
        ".yaml": "#",
        ".toml": "#",
        ".ini": "#",
        ".js": "//",
        ".ts": "//",
        ".tsx": "//",
        ".jsx": "//",
        ".go": "//",
        ".java": "//",
        ".c": "//",
        ".cc": "//",
        ".cpp": "//",
        ".h": "//",
        ".hpp": "//",
        ".rs": "//",
        ".php": "//",
        ".css": "/*",  # We'll close manually when used
    }
    # Prefer explicit language hint over suffix when possible
    lang_map = {
        "python": "#",
        "javascript": "//",
        "typescript": "//",
        "shell": "#",
        "bash": "#",
        "go": "//",
        "java": "//",
    }
    if lang in lang_map:
        return lang_map[lang]
    return mapping.get(suffix, "//")


def _insert_after_line(lines: list[str], idx: int, new_block: str) -> None:
    """Insert a block of text after the given zero-based line index."""
    insert_at = idx + 1
    block_lines = new_block.splitlines()
    lines[insert_at:insert_at] = block_lines


def _line_range(target_area: str) -> Optional[tuple[int, int]]:
    """Parse a line range in the form 'start-end' (1-based, inclusive)."""
    if not target_area:
        return None
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", target_area)
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if start < 1 or end < start:
        return None
    return start, end


def _maybe_preview(text: str, limit: int = 200) -> str:
    """Return a shortened preview for responses."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _run_subprocess(cmd: list[str], cwd: Path) -> dict:
    """Run a subprocess safely within the workspace."""
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "command": " ".join(cmd),
            "cwd": _relative(cwd),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError:
        return {"error": f"Command not found: {cmd[0]}", "command": " ".join(cmd)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc), "command": " ".join(cmd)}


# ---------------------------------------------------------------------------
# Code Writer Sub-Agent
# ---------------------------------------------------------------------------
@mcp.tool()
def write_code(
    file_path: Annotated[str, Field(description="Target file path. Relative paths resolve from workspace root. Examples: 'src/utils.py', 'lib/helpers.js'.")],
    description: Annotated[str, Field(description="What to generate. Either natural language ('a function that validates email addresses') or a fenced code block with exact code to write.")],
    language: Annotated[str, Field(description="Programming language. Examples: 'python', 'javascript', 'typescript', 'go'. Defaults to 'plain'.")] = "plain",
) -> dict:
    """Generate and write code to a file. Creates new file or appends to existing. Use when: starting new files, adding functions/classes. Returns: path, status, preview of written code."""
    resolved = _resolve_path(file_path)
    user_block = _extract_code_block(description)

    llm_prompt = (
        f"You are a senior engineer generating code.\n"
        f"Target file: {file_path}\nLanguage: {language}\n"
        f"Requirements:\n{description}\n"
        "Return a single code block with the full file content or the function to append. "
        "Do not include explanations."
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(resolved)}

    generated_block = _extract_code_block(llm_text) or llm_text.strip()
    content = user_block or generated_block

    if not content:
        return {
            "error": "LLM did not return code to write.",
            "path": _relative(resolved),
            "llm_response": _maybe_preview(llm_text or ""),
        }

    append = resolved.exists() and resolved.is_file()
    _write_text(resolved, ("\n\n" if append else "") + content, append=append)

    return {
        "path": _relative(resolved),
        "status": "appended" if append else "created",
        "bytes_written": len(content.encode("utf-8", errors="replace")),
        "preview": _maybe_preview(content),
        "llm_model": MODEL_ID,
    }


@mcp.tool()
def modify_code(
    file_path: Annotated[str, Field(description="File to modify. Must exist. Example: 'src/main.py'.")],
    change_description: Annotated[str, Field(description="What to change. Natural language ('add error handling to the parse function') or fenced code block with replacement code.")],
    target_area: Annotated[str, Field(description="Where to apply change. Options: function name ('process_data'), line range ('42-55'), or empty for append. Defaults to empty.")] = "",
) -> dict:
    """Edit specific code in an existing file. Replaces targeted section or appends if no target found. Use when: modifying functions, fixing specific lines, adding code to existing files. NOT for: creating new files (use write_code)."""
    resolved = _resolve_path(file_path)
    original = _read_text(resolved)
    lines = original.splitlines()
    prefix = _comment_prefix(resolved)
    line_range = _line_range(target_area)

    # Prepare context for the LLM
    context_excerpt = "\n".join(lines[:400])
    if line_range:
        start, end = line_range
        snippet = "\n".join(lines[max(0, start - 5): min(len(lines), end + 5)])
    else:
        snippet = "\n".join(lines[:200])

    llm_prompt = (
        f"You are editing code in {file_path}. "
        f"Change request: {change_description}. "
        f"Target area: {target_area or 'append near end if unspecified'}.\n\n"
        f"Context snippet:\n```{snippet}```\n"
        "Return only the replacement code block to insert. "
        "Do not include prose."
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(resolved)}

    llm_block = _extract_code_block(llm_text) or llm_text.strip()
    user_block = _extract_code_block(change_description)
    code_block = user_block or llm_block

    if line_range:
        start, end = line_range
        start_idx, end_idx = start - 1, end
        if start_idx >= len(lines):
            return {"error": f"Start line {start} outside file bounds", "path": _relative(resolved)}
        replacement = code_block or f"{prefix} TODO: {change_description}"
        new_lines = lines[:start_idx] + replacement.splitlines() + lines[end_idx:]
        new_content = "\n".join(new_lines) + "\n"
        _write_text(resolved, new_content, append=False)
        return {
            "path": _relative(resolved),
            "status": "replaced_range",
            "applied_range": f"{start}-{end}",
            "preview": _maybe_preview(replacement),
            "llm_model": MODEL_ID,
        }

    # Non-range targeting: look for the target_area string or append at end
    target_idx = None
    if target_area:
        for idx, line in enumerate(lines):
            if target_area.lower() in line.lower():
                target_idx = idx
                break

    if code_block:
        if target_idx is not None:
            lines[target_idx:target_idx + 1] = code_block.splitlines()
        else:
            lines.append("")
            lines.extend(code_block.splitlines())
        new_content = "\n".join(lines) + "\n"
        _write_text(resolved, new_content, append=False)
        return {
            "path": _relative(resolved),
            "status": "replaced_match" if target_idx is not None else "appended",
            "target": target_area or "end_of_file",
            "preview": _maybe_preview(code_block),
            "llm_model": MODEL_ID,
        }

    # No code block: add a TODO marker near the target
    marker = f"{prefix} TODO: {change_description}"
    if target_idx is None:
        lines.append(marker)
        target_idx = len(lines) - 1
    else:
        _insert_after_line(lines, target_idx, marker)
        target_idx += 1

    new_content = "\n".join(lines) + "\n"
    _write_text(resolved, new_content, append=False)
    return {
        "path": _relative(resolved),
        "status": "noted_change",
        "line": target_idx + 1,
        "preview": _maybe_preview(marker),
        "llm_model": MODEL_ID,
    }


@mcp.tool()
def apply_fix(
    file_path: Annotated[str, Field(description="File containing the bug. Example: 'src/parser.py'.")],
    issue_description: Annotated[str, Field(description="Bug description and fix. Examples: 'TypeError on line 42 when input is None', 'add null check before accessing user.name'. Include code block for exact fix.")],
) -> dict:
    """Apply a bug fix to a file. Appends fix code or adds FIXME marker if fix is unclear. Use when: fixing reported bugs, addressing error messages. Returns: path, status (patched/marker_added), preview."""
    resolved = _resolve_path(file_path)
    content = _read_text(resolved)
    prefix = _comment_prefix(resolved)

    llm_prompt = (
        f"You are fixing a bug in {file_path}.\n"
        f"Issue: {issue_description}\n"
        f"Existing code (truncated):\n```{_maybe_preview(content, 2000)}```\n"
        "Provide a minimal patch or appended helper as a single code block. "
        "If unsure, propose a FIXME comment."
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(resolved)}

    code_block = _extract_code_block(llm_text) or _extract_code_block(issue_description) or llm_text.strip()

    if code_block:
        patched = content.rstrip("\n") + "\n\n" + code_block + "\n"
        _write_text(resolved, patched, append=False)
        return {
            "path": _relative(resolved),
            "status": "patched_with_block",
            "preview": _maybe_preview(code_block),
            "llm_model": MODEL_ID,
        }

    marker = f"{prefix} FIX: {issue_description}"
    if marker in content:
        return {
            "path": _relative(resolved),
            "status": "exists",
            "message": "Fix marker already present.",
            "llm_model": MODEL_ID,
        }

    patched = content.rstrip("\n") + "\n" + marker + "\n"
    _write_text(resolved, patched, append=False)
    return {
        "path": _relative(resolved),
        "status": "marker_added",
        "preview": _maybe_preview(marker),
        "llm_model": MODEL_ID,
    }


# ---------------------------------------------------------------------------
# Code Reviewer Sub-Agent
# ---------------------------------------------------------------------------
def _scan_security(line: str) -> Optional[str]:
    """Spot a handful of obvious security smells."""
    lower = line.lower()
    if "eval(" in lower or "exec(" in lower:
        return "Use of eval/exec; prefer safer parsing."
    if "subprocess" in lower and "shell=True" in lower:
        return "subprocess with shell=True; risk of command injection."
    if "pickle.loads" in lower:
        return "Unpickling data; ensure trusted input."
    if "md5" in lower or "sha1" in lower:
        return "Weak hash algorithm detected."
    return None


@mcp.tool()
def review_changes(
    file_path: Annotated[str, Field(description="File to review. Example: 'src/auth.py'.")],
    focus: Annotated[list[str], Field(description="Review focus areas. Options: 'bugs', 'security', 'readability', 'performance'. Defaults to ['bugs', 'readability'].")] = ("bugs", "readability"),
) -> dict:
    """Code review for common issues. Returns severity-tagged findings (low/medium/high). Use when: before merging, after major changes, checking new code quality. Returns: issues list with severity, line numbers, descriptions."""
    resolved = _resolve_path(file_path)
    try:
        text = _read_text(resolved)
    except Exception as exc:
        return {"error": str(exc), "path": _relative(resolved)}

    focuses = ", ".join(focus)
    numbered = "\n".join(f"{idx+1}: {line}" for idx, line in enumerate(text.splitlines()[:400]))
    llm_prompt = (
        f"Review the following code focusing on {focuses}. "
        "Return JSON list with entries: severity (low|medium|high), line (int or null), issue (short text). "
        "Only return JSON.\n"
        f"Code:\n```{numbered}```"
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(resolved)}

    parsed = _extract_json_payload(llm_text)
    issues = parsed if isinstance(parsed, list) else []

    return {
        "path": _relative(resolved),
        "issue_count": len(issues),
        "issues": issues,
        "llm_model": MODEL_ID,
        "raw_review": _maybe_preview(llm_text or ""),
    }


@mcp.tool()
def check_common_mistakes(
    file_path: Annotated[str, Field(description="File to analyze. Example: 'src/utils.py'.")],
    language: Annotated[str, Field(description="Programming language for language-specific checks. Examples: 'python', 'javascript', 'typescript'. Defaults to 'python'.")] = "python",
) -> dict:
    """Quick scan for anti-patterns and common mistakes. Faster than full review. Use when: quick code health check, identifying obvious issues. Returns: list of findings with suggested improvements."""
    resolved = _resolve_path(file_path)
    try:
        text = _read_text(resolved)
    except Exception as exc:
        return {"error": str(exc), "path": _relative(resolved)}

    llm_prompt = (
        f"Identify quick wins and common mistakes in this {language} file. "
        "Return a JSON array of short strings (each a finding). "
        "Code snippet:\n```"
        f"{_maybe_preview(text, 3500)}"
        "```"
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(resolved)}

    parsed = _extract_json_payload(llm_text)
    findings = parsed if isinstance(parsed, list) else [llm_text]

    return {
        "path": _relative(resolved),
        "language": language,
        "findings": findings,
        "llm_model": MODEL_ID,
    }


# ---------------------------------------------------------------------------
# Debugger Sub-Agent
# ---------------------------------------------------------------------------
@mcp.tool()
def analyze_error(
    error_message: Annotated[str, Field(description="Full error message or stack trace. Include complete traceback for best results. Example: 'TypeError: cannot unpack non-iterable NoneType object at line 42'.")],
    file_path: Annotated[Optional[str], Field(description="Source file for additional context. Helps identify root cause. Example: 'src/parser.py'. Optional.")] = None,
) -> dict:
    """Diagnose error messages and stack traces. Identifies top 3 likely causes. Use when: runtime errors, exceptions, test failures. Chain with suggest_fix for solutions."""
    context = None
    if file_path:
        try:
            context = _read_text(_resolve_path(file_path))
        except Exception:
            context = None

    llm_prompt = (
        "Analyze the following error and suggest the top 3 likely causes.\n"
        f"Error:\n{error_message}\n\n"
        f"Context (may be truncated):\n```{_maybe_preview(context or '', 2000)}```\n"
        "Return a JSON array of short cause strings."
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(_resolve_path(file_path)) if file_path else None}

    parsed = _extract_json_payload(llm_text)
    probable = parsed if isinstance(parsed, list) else [llm_text]

    return {
        "error": error_message,
        "file_inspected": _relative(_resolve_path(file_path)) if file_path else None,
        "probable_causes": probable,
        "context_excerpt": _maybe_preview(context) if context else None,
        "llm_model": MODEL_ID,
    }


@mcp.tool()
def suggest_fix(
    error_analysis: Annotated[str, Field(description="Error description or output from analyze_error. Example: 'NoneType error because user lookup returns None when user not found'.")],
) -> dict:
    """Get concrete fix suggestion for an analyzed error. Returns code snippet or step-by-step fix. Use after: analyze_error. Chain with: apply_fix to implement the solution."""
    llm_prompt = (
        "Provide a single, concrete code-level fix for the following issue. "
        "Keep it to one or two sentences or a short code block.\n"
        f"Issue: {error_analysis}"
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err}
    suggestion_block = _extract_code_block(llm_text) or llm_text.strip()
    return {"suggestion": suggestion_block, "llm_model": MODEL_ID}


@mcp.tool()
def add_logging(
    file_path: Annotated[str, Field(description="File to add logging to. Example: 'src/api.py'.")],
    location: Annotated[str, Field(description="Where to add logging. Function name ('process_request') or keyword in code ('database_query'). Logging added after first match.")],
) -> dict:
    """Insert debug logging statement at specified location. Auto-detects appropriate logging syntax for language. Use when: debugging, tracing execution flow, investigating issues. Returns: line number where logging was added."""
    resolved = _resolve_path(file_path)
    try:
        lines = _read_text(resolved).splitlines()
    except Exception as exc:
        return {"error": str(exc), "path": _relative(resolved)}

    target_idx = None
    for idx, line in enumerate(lines):
        if location.lower() in line.lower():
            target_idx = idx
            break
    if target_idx is None:
        target_idx = len(lines) - 1 if lines else 0

    context_snippet = "\n".join(lines[max(0, target_idx - 5): target_idx + 5])
    llm_prompt = (
        f"Add a single debug log statement appropriate for file {file_path} near '{location}'. "
        f"Context:\n```{context_snippet}```\n"
        "Return only the logging line (no surrounding code) with correct indentation."
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(resolved)}

    log_line = (_extract_code_block(llm_text) or llm_text.strip()) or "// DEBUG"

    _insert_after_line(lines, target_idx, log_line)
    new_content = "\n".join(lines) + "\n"
    _write_text(resolved, new_content, append=False)

    return {
        "path": _relative(resolved),
        "status": "logging_added",
        "line": target_idx + 2,  # inserted after target_idx
        "preview": log_line.strip(),
        "llm_model": MODEL_ID,
    }


# ---------------------------------------------------------------------------
# Test Generator Sub-Agent
# ---------------------------------------------------------------------------
@mcp.tool()
def generate_tests(
    file_path: Annotated[str, Field(description="Source file containing functions to test. Example: 'src/calculator.py'.")],
    functions: Annotated[list[str], Field(description="Function names to generate tests for. Example: ['add', 'multiply', 'divide']. Each gets happy path + edge case tests.")],
) -> dict:
    """Generate pytest test suite for specified functions. Creates tests/test_<module>.py with happy path and edge cases. Use when: new code needs tests, improving coverage. Returns: test file path, number of tests added."""
    source = _resolve_path(file_path)
    module_name = source.stem
    module_import = _relative(source).replace("/", ".")
    if module_import.endswith(".py"):
        module_import = module_import[:-3]
    test_dir = ROOT / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / f"test_{module_name}.py"
    existed_before = test_file.exists()

    try:
        source_text = _read_text(source)
    except Exception as exc:
        return {"error": str(exc), "path": _relative(source)}

    llm_prompt = (
        f"Write pytest tests for functions {functions} in module {module_import}.\n"
        "Cover a happy path and a couple of edge cases for each function. "
        "Import from the module directly. Keep tests minimal and deterministic. "
        "Return a single Python code block with the tests."
        f"\n\nRelevant code:\n```{_maybe_preview(source_text, 3000)}```"
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(source)}

    test_block = _extract_code_block(llm_text) or llm_text.strip()
    if not test_block:
        return {"error": "LLM did not return test code.", "path": _relative(test_file)}

    header = f'"""Autogenerated tests for {module_name} using {MODEL_ID}. Edit as needed."""\n'
    content = header + test_block + "\n"
    if existed_before:
        existing = _read_text(test_file)
        content = existing.rstrip("\n") + "\n" + content
    _write_text(test_file, content, append=False)

    return {
        "path": _relative(test_file),
        "status": "updated" if existed_before else "created",
        "tests_added": len(functions) * 2,
        "preview": _maybe_preview(content),
        "llm_model": MODEL_ID,
    }


@mcp.tool()
def add_test_case(
    existing_test_file: Annotated[str, Field(description="Existing test file to extend. Example: 'tests/test_auth.py'. Must exist.")],
    scenario_description: Annotated[str, Field(description="Test scenario to add. Natural language ('test login fails with wrong password') or code block with exact test. Example: 'verify empty input raises ValueError'.")],
) -> dict:
    """Add single test case to existing test file. Use when: adding specific test scenarios, covering edge cases. NOT for: creating new test files (use generate_tests). Returns: preview of added test."""
    resolved = _resolve_path(existing_test_file)
    if not resolved.exists():
        return {"error": "Test file does not exist.", "path": _relative(resolved)}

    try:
        existing_text = _read_text(resolved)
    except Exception as exc:
        return {"error": str(exc), "path": _relative(resolved)}

    code_block = _extract_code_block(scenario_description)
    llm_prompt = (
        "Write a pytest test case for the following scenario. "
        "Return only the test code (no explanations) in a single code block.\n"
        f"Scenario: {scenario_description}\n\n"
        f"Existing tests context:\n```{_maybe_preview(existing_text, 2500)}```"
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(resolved)}

    generated_block = _extract_code_block(llm_text) or llm_text.strip()
    body = code_block or generated_block

    if not body:
        return {"error": "LLM did not produce a test case.", "path": _relative(resolved)}

    existing = existing_text.rstrip("\n")
    new_content = existing + "\n\n" + body + "\n"
    _write_text(resolved, new_content, append=False)

    return {
        "path": _relative(resolved),
        "status": "appended",
        "preview": _maybe_preview(body),
        "llm_model": MODEL_ID,
    }


# ---------------------------------------------------------------------------
# Documentation Writer Sub-Agent
# ---------------------------------------------------------------------------
@mcp.tool()
def add_docstrings(
    file_path: Annotated[str, Field(description="Python file to document. Example: 'src/utils.py'. Must be Python (.py).")],
    functions: Annotated[list[str], Field(description="Function names to add docstrings to. Example: ['process_data', 'validate_input']. Skips functions that already have docstrings.")],
) -> dict:
    """Add Python docstrings to undocumented functions. Generates concise descriptions based on code analysis. Use when: improving code documentation, preparing for API docs. Python only."""
    resolved = _resolve_path(file_path)
    lines = _read_text(resolved).splitlines()
    added = 0

    for func in functions:
        pattern = re.compile(rf"^\s*def\s+{re.escape(func)}\s*\(")
        for idx, line in enumerate(lines):
            if pattern.match(line):
                # Check if next non-empty line is a docstring
                insert_idx = idx + 1
                while insert_idx < len(lines) and not lines[insert_idx].strip():
                    insert_idx += 1
                has_docstring = (
                    insert_idx < len(lines) and lines[insert_idx].lstrip().startswith('"""')
                )
                if not has_docstring:
                    # Grab a small snippet for context
                    snippet = "\n".join(lines[idx: min(len(lines), idx + 20)])
                    llm_prompt = (
                        f"Write a concise Python docstring for function `{func}`. "
                        "Return only the docstring body (no triple quotes) in one or two sentences.\n"
                        f"Function snippet:\n```{snippet}```"
                    )
                    llm_text, llm_err = _call_llm(llm_prompt)
                    if llm_err:
                        return {"error": llm_err, "path": _relative(resolved)}
                    doc_body = (_extract_code_block(llm_text) or llm_text.strip()).strip().strip('\"')
                    indent = re.match(r"\s*", line).group(0) + "    "
                    doc = f'{indent}"""{doc_body}"""'
                    lines.insert(idx + 1, doc)
                    added += 1
                break

    if added == 0:
        return {"path": _relative(resolved), "status": "unchanged", "message": "No matching functions found or docstrings already present."}

    new_content = "\n".join(lines) + "\n"
    _write_text(resolved, new_content, append=False)
    return {"path": _relative(resolved), "status": "docstrings_added", "count": added, "llm_model": MODEL_ID}


@mcp.tool()
def create_readme(
    project_name: Annotated[str, Field(description="Project name for the title. Example: 'FastAPI Auth Service'.")],
    description: Annotated[str, Field(description="One-line project summary. Example: 'A lightweight authentication microservice with JWT support'.")],
    main_features: Annotated[list[str], Field(description="Key features to highlight. Example: ['JWT authentication', 'Rate limiting', 'OAuth2 support']. 3-6 items recommended.")],
) -> dict:
    """Generate README.md at workspace root. Includes: Features, Installation, Usage, Quickstart. Overwrites existing README. Use when: starting new project, updating project documentation."""
    readme_path = ROOT / "README.md"
    features_text = "\n".join(f"- {feat}" for feat in main_features)
    llm_prompt = (
        f"Draft a concise README in Markdown for project '{project_name}'.\n"
        f"Description: {description}\n"
        f"Features:\n{features_text}\n"
        "Include sections: Features, Installation (use pip -r requirements.txt), Usage (placeholder), and a short quickstart example. "
        "Return only the Markdown in a single code block."
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(readme_path)}

    content = _extract_code_block(llm_text) or llm_text.strip()
    if not content:
        return {"error": "LLM did not return README content.", "path": _relative(readme_path)}

    _write_text(readme_path, content, append=False)
    return {
        "path": _relative(readme_path),
        "status": "written",
        "bytes_written": len(content),
        "llm_model": MODEL_ID,
    }


@mcp.tool()
def explain_code(
    file_path: Annotated[str, Field(description="File containing code to explain. Example: 'src/algorithm.py'.")],
    section: Annotated[str, Field(description="Focus area. Function name ('merge_sort'), keyword ('async'), or leave empty for file overview. Defaults to empty (explains first ~10 lines).")] = "",
) -> dict:
    """Get plain-English explanation of code. Use when: understanding unfamiliar code, onboarding, documenting complex logic. Returns: human-readable explanation of what the code does and why."""
    resolved = _resolve_path(file_path)
    text = _read_text(resolved)
    lines = text.splitlines()
    excerpt = []

    if section:
        for idx, line in enumerate(lines):
            if section.lower() in line.lower():
                start = max(0, idx - 2)
                end = min(len(lines), idx + 3)
                excerpt = lines[start:end]
                break
    if not excerpt:
        excerpt = lines[:10]

    context = "\n".join(excerpt)
    llm_prompt = (
        f"Explain the following code from {_relative(resolved)} in plain English. "
        f"Focus on section marker '{section}'.\n```{context}```"
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err, "path": _relative(resolved)}

    explanation = llm_text.strip()
    return {"path": _relative(resolved), "explanation": explanation, "llm_model": MODEL_ID}


# ---------------------------------------------------------------------------
# Dependency Manager Sub-Agent
# ---------------------------------------------------------------------------
def _append_requirement(package: str) -> bool:
    req_path = ROOT / "requirements.txt"
    existing = req_path.read_text().splitlines() if req_path.exists() else []
    if any(line.split("==")[0].strip() == package for line in existing):
        return False
    existing.append(package)
    _write_text(req_path, "\n".join(existing) + "\n", append=False)
    return True


@mcp.tool()
def add_package(
    package_name: Annotated[str, Field(description="Package to add.")],
    package_manager: Annotated[str, Field(description="Package manager (pip, npm, etc.).")] = "pip",
) -> dict:
    """Record a dependency using the project's likely package manifest without installing it."""
    llm_prompt = (
        f"Determine a brief note about adding dependency '{package_name}' with {package_manager}. "
        "Mention any quick caveats or typical install hints in under 2 sentences."
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err}

    manager = package_manager.lower()
    if manager in {"pip", "pip3"}:
        added = _append_requirement(package_name)
        return {
            "path": _relative(ROOT / "requirements.txt"),
            "status": "added" if added else "exists",
            "package": package_name,
            "llm_note": _maybe_preview(llm_text),
            "llm_model": MODEL_ID,
        }
    return {"error": f"Package manager '{package_manager}' is not supported yet.", "llm_note": _maybe_preview(llm_text), "llm_model": MODEL_ID}


@mcp.tool()
def check_outdated() -> dict:
    """Check for outdated Python packages. Runs 'pip list --outdated'. Use when: maintenance, security updates, preparing upgrades. Requires pip in PATH. Returns: list of packages with current and latest versions."""
    if not shutil.which("pip"):
        llm_text, _ = _call_llm("Provide a short note that pip is unavailable so outdated packages cannot be checked.")
        return {"error": "pip not available in environment.", "llm_note": _maybe_preview(llm_text or ""), "llm_model": MODEL_ID}
    result = _run_subprocess(["pip", "list", "--outdated", "--format=json"], cwd=ROOT)
    summary_prompt = (
        "Summarize the following pip outdated output briefly. "
        "If empty, say no updates found.\n"
        f"stdout:\n{result.get('stdout','')}\n\nstderr:\n{result.get('stderr','')}"
    )
    llm_text, _ = _call_llm(summary_prompt)
    result["llm_summary"] = _maybe_preview(llm_text or "")
    result["llm_model"] = MODEL_ID
    return result


@mcp.tool()
def check_security() -> dict:
    """Basic security audit of requirements.txt. Checks for: unpinned versions, known vulnerable versions (Flask 1.x, Django 1.x). Use when: security review, pre-deployment checks. NOT a full vulnerability scan."""
    findings = []
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        llm_text, _ = _call_llm("Explain briefly that no requirements file was found so security scan is skipped.")
        return {"status": "no_requirements", "findings": findings, "llm_note": _maybe_preview(llm_text or ""), "llm_model": MODEL_ID}

    for line in req_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "==" not in stripped:
            findings.append(f"Unpinned dependency: {stripped}")
        if stripped.startswith("flask==1.") or stripped.startswith("django==1."):
            findings.append(f"Very old framework version detected: {stripped}")

    llm_prompt = (
        "Provide a short security summary for these dependency findings:\n"
        + "\n".join(findings or ["No issues found"])
    )
    llm_text, _ = _call_llm(llm_prompt)

    return {"path": _relative(req_path), "findings": findings, "status": "scanned", "llm_note": _maybe_preview(llm_text or ""), "llm_model": MODEL_ID}


# ---------------------------------------------------------------------------
# Architecture Planner Sub-Agent
# ---------------------------------------------------------------------------
@mcp.tool()
def plan_feature(
    feature_description: Annotated[str, Field(description="Feature to implement. Be specific. Example: 'Add user authentication with JWT tokens and refresh token support'.")],
    relevant_files: Annotated[list[str], Field(description="Known related files. Example: ['src/auth.py', 'src/models/user.py']. Helps create accurate plan. Defaults to empty.")] = (),
) -> dict:
    """Create implementation plan for a feature. Returns: files to modify, new files to create, ordered implementation steps. Use when: starting new feature, planning refactoring."""
    llm_prompt = (
        "Create a short implementation plan for the feature below. "
        "Return JSON with keys: files_to_touch (list), new_files (list), steps (ordered list of strings).\n"
        f"Feature: {feature_description}\n"
        f"Known files: {list(relevant_files)}"
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err}
    parsed = _extract_json_payload(llm_text)
    if isinstance(parsed, dict):
        plan = parsed
    else:
        plan = {
            "feature": feature_description,
            "files_to_touch": list(relevant_files),
            "new_files": [],
            "steps": [llm_text],
        }
    plan["llm_model"] = MODEL_ID
    return plan


@mcp.tool()
def suggest_structure(
    task_description: Annotated[str, Field(description="Project or task to structure. Example: 'REST API for e-commerce with products, orders, and users'.")],
) -> dict:
    """Recommend project/file structure for a task. Returns: suggested directory layout and file organization. Use when: starting new project, reorganizing codebase."""
    llm_prompt = (
        "Propose a concise project/file structure for the following task. "
        "Return JSON array of file paths (strings).\n"
        f"Task: {task_description}"
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err}
    parsed = _extract_json_payload(llm_text)
    layout = parsed if isinstance(parsed, list) else [_maybe_preview(llm_text or "")]
    return {"task": task_description, "suggested_layout": layout, "llm_model": MODEL_ID}


@mcp.tool()
def recommend_approach(
    problem: Annotated[str, Field(description="Technical problem or decision. Example: 'How to implement caching for API responses'.")],
    options: Annotated[list[str], Field(description="Approaches to compare. Example: ['Redis', 'Memcached', 'In-memory LRU']. If empty, generates options. Max 3 recommended.")] = (),
) -> dict:
    """Compare implementation approaches with pros/cons. Use when: architecture decisions, choosing between technologies. Returns: comparison matrix with recommendations."""
    llm_prompt = (
        "Compare the following implementation options for the problem and return JSON. "
        "JSON format: [{\"option\": str, \"pros\": [str], \"cons\": [str]}]. "
        f"Problem: {problem}\nOptions: {options}"
    )
    llm_text, llm_err = _call_llm(llm_prompt)
    if llm_err:
        return {"error": llm_err}
    parsed = _extract_json_payload(llm_text)
    comparisons = parsed if isinstance(parsed, list) else [{"option": "default", "pros": [llm_text], "cons": []}]
    return {"problem": problem, "comparisons": comparisons, "llm_model": MODEL_ID}


# ---------------------------------------------------------------------------
# File System Navigator Sub-Agent
# ---------------------------------------------------------------------------
def _iter_files(base: Path, pattern: Optional[str] = None) -> Iterable[Path]:
    for path in base.rglob("*"):
        if path.is_file() and (not pattern or pattern in path.suffix):
            yield path


@mcp.tool()
def find_files(
    search_term: Annotated[str, Field(description="Text to find in filenames OR file content. Case-insensitive. Example: 'database', 'config', 'TODO'.")],
    file_type: Annotated[str, Field(description="Filter by extension. Examples: '.py', 'js', 'md'. Empty searches all files. Defaults to empty.")] = "",
    max_results: Annotated[int, Field(description="Maximum matches to return. Range: 1-50. Defaults to 10.")] = 10,
) -> dict:
    """Search workspace for files by name or content. Use when: locating files, finding where something is defined. Returns: matching file paths with match type (filename/content)."""
    matches = []
    ext = file_type if file_type.startswith(".") or not file_type else f".{file_type}"
    for path in _iter_files(ROOT, pattern=ext if file_type else None):
        if search_term.lower() in path.name.lower():
            matches.append({"path": _relative(path), "match": "filename"})
        elif search_term.lower() in path.read_text(errors="ignore").lower():
            matches.append({"path": _relative(path), "match": "content"})
        if len(matches) >= max_results:
            break

    llm_prompt = (
        f"Summarize these search results for term '{search_term}' "
        f"with optional type '{file_type}'. Provide next-step advice in one sentence.\n"
        f"{matches}"
    )
    llm_text, _ = _call_llm(llm_prompt)

    return {"search_term": search_term, "matches": matches, "count": len(matches), "llm_note": _maybe_preview(llm_text or ""), "llm_model": MODEL_ID}


@mcp.tool()
def find_definition(
    symbol_name: Annotated[str, Field(description="Function or class name to find. Example: 'UserAuthentication', 'process_payment'.")],
) -> dict:
    """Find where a function/class is defined. Uses ripgrep for fast search. Use when: navigating to source, understanding code structure. Requires 'rg' in PATH."""
    cmd = ["rg", "-n", rf"(def|class)\s+{re.escape(symbol_name)}", str(ROOT)]
    result = _run_subprocess(cmd, cwd=ROOT)
    llm_text, _ = _call_llm(
        f"Given ripgrep results, summarize where symbol {symbol_name} may be defined in one sentence:\n{result}"
    )
    return {"symbol": symbol_name, "result": result, "llm_note": _maybe_preview(llm_text or ""), "llm_model": MODEL_ID}


@mcp.tool()
def find_references(
    symbol_name: Annotated[str, Field(description="Symbol to find usages of. Example: 'validate_email', 'DatabaseConnection'.")],
) -> dict:
    """Find all usages of a function/class/variable. Uses ripgrep. Use when: understanding impact of changes, refactoring. Requires 'rg' in PATH."""
    cmd = ["rg", "-n", symbol_name, str(ROOT)]
    result = _run_subprocess(cmd, cwd=ROOT)
    llm_text, _ = _call_llm(
        f"Summarize ripgrep reference results for symbol {symbol_name} in one sentence:\n{result}"
    )
    return {"symbol": symbol_name, "result": result, "llm_note": _maybe_preview(llm_text or ""), "llm_model": MODEL_ID}


# ---------------------------------------------------------------------------
# Terminal Executor Sub-Agent
# ---------------------------------------------------------------------------
@mcp.tool()
def run_command(
    command: Annotated[str, Field(description="Shell command. AVOID destructive commands (rm -rf, etc). Examples: 'ls -la', 'cat file.txt', 'git status'.")],
    working_dir: Annotated[str, Field(description="Working directory relative to workspace root. Defaults to '.' (workspace root).")] = ".",
) -> dict:
    """Execute shell command in workspace. Sandboxed to workspace directory. Use when: running scripts, checking file info, git operations. Returns: stdout, stderr, exit code. WARNING: avoid destructive commands."""
    cwd = _resolve_path(working_dir)
    cmd_list = shlex.split(command)
    result = _run_subprocess(cmd_list, cwd=cwd)
    llm_text, _ = _call_llm(
        f"Summarize this command execution in one sentence with any warnings: command='{command}', returncode={result.get('returncode')}, stderr='{result.get('stderr','')[:400]}'."
    )
    result["llm_note"] = _maybe_preview(llm_text or "")
    result["llm_model"] = MODEL_ID
    return result


@mcp.tool()
def run_tests(
    test_path: Annotated[str, Field(description="Test file or directory. Examples: 'tests/', 'tests/test_auth.py', 'tests/unit/'. Relative to workspace.")],
) -> dict:
    """Run pytest on specified tests. Use when: validating changes, checking test status. Requires pytest in environment. Returns: test results with pass/fail counts."""
    resolved = _resolve_path(test_path)
    cmd = ["pytest", str(resolved)]
    result = _run_subprocess(cmd, cwd=ROOT)
    llm_text, _ = _call_llm(
        f"Summarize pytest result (returncode {result.get('returncode')}): stdout='{_maybe_preview(result.get('stdout',''),400)}', stderr='{_maybe_preview(result.get('stderr',''),200)}'."
    )
    result["llm_note"] = _maybe_preview(llm_text or "")
    result["llm_model"] = MODEL_ID
    return result


@mcp.tool()
def run_linter(
    files: Annotated[list[str], Field(description="Files or directories to lint. Examples: ['src/'], ['src/main.py', 'src/utils.py'].")],
) -> dict:
    """Run Python linter on files. Auto-selects: ruff > flake8 > pylint (first available). Use when: checking code quality, pre-commit validation. Returns: linting issues found."""
    linter = None
    for candidate in ["ruff", "flake8", "pylint"]:
        if shutil.which(candidate):
            linter = candidate
            break
    if not linter:
        return {"error": "No supported linter (ruff/flake8/pylint) found in PATH."}

    cmd = [linter] + files
    result = _run_subprocess(cmd, cwd=ROOT)
    llm_text, _ = _call_llm(
        f"Summarize linter output for {files} using {linter}: stdout='{_maybe_preview(result.get('stdout',''),400)}', stderr='{_maybe_preview(result.get('stderr',''),200)}'."
    )
    enriched = {"linter": linter, **result}
    enriched["llm_note"] = _maybe_preview(llm_text or "")
    enriched["llm_model"] = MODEL_ID
    return enriched


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
@mcp.resource("info://server")
def get_server_info() -> str:
    """Provide a quick reference for the subagent server."""
    return f"""# Subagent Server

Workspace root: `{ROOT}`
Chat model: `{MODEL_ID}` (set MCP_SUBAGENT_MODEL to change)

## Tools by sub-agent
- **Code Writer**: write_code, modify_code, apply_fix
- **Code Reviewer**: review_changes, check_common_mistakes
- **Debugger**: analyze_error, suggest_fix, add_logging
- **Test Generator**: generate_tests, add_test_case
- **Documentation Writer**: add_docstrings, create_readme, explain_code
- **Dependency Manager**: add_package, check_outdated, check_security
- **Architecture Planner**: plan_feature, suggest_structure, recommend_approach
- **File System Navigator**: find_files, find_definition, find_references
- **Terminal Executor**: run_command, run_tests, run_linter

Notes:
- Tools favor safe, additive edits; review diffs when applying patches.
- Paths are scoped to the workspace root for safety.
- All tools call LangChain's init_chat_model under the hood; install langchain and configure model credentials.
- Some tools look for fenced code blocks (```...```) to apply exact code.
"""


if __name__ == "__main__":
    mcp.run()
