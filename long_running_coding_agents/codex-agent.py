#!/usr/bin/env python3
"""
codex-agent.py

A long-running, offline-first coding agent built on LangChain's `create_deep_agent`.

Goals
- Multi-turn CLI loop with Ctrl+C interrupts to inject new instructions mid-run.
- Offline filesystem tools (read/write/edit/search) with disk-backed memory.
- Summarization middleware with customizable prompt to manage very long contexts.
- Subagent spawning for isolating subtasks.
- Minimal CLI; most configuration lives in this file for easy tweaking.

Dependencies (expected to be available in your environment):
  langchain, langgraph, python-dotenv
"""

import argparse
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Load .env from the current working directory.
load_dotenv()

# =========================
# User-tunable defaults
# =========================

# Preferred chat model for the main agent.
MODEL_NAME = os.environ.get("CODEX_MODEL", "gpt-5-nano")
MODEL_PROVIDER = os.environ.get("CODEX_MODEL_PROVIDER")  # e.g., "openai", "anthropic"
MODEL_KWARGS: Dict[str, object] = {"temperature": 0.2}

# Faster/cheaper model for summarization. Disable with --no-summary.
SUMMARY_MODEL = os.environ.get("CODEX_SUMMARY_MODEL", "gpt-5-nano")
SUMMARY_PROMPT = os.environ.get(
    "CODEX_SUMMARY_PROMPT",
    "You are a concise note-taker. Summarize the conversation so far, focusing on "
    "decisions, open questions, test results, and file paths touched. Keep it short "
    "but sufficient to resume later. Do not lose TODOs.",
)

# System prompt appended to the built-in create_deep_agent instructions.
SYSTEM_PROMPT = """
You are an expert software engineer working on long-running, offline coding tasks.
- You have NO internet; rely only on local files and user-provided context.
- Use filesystem tools to read/write/edit/search code.
- Persist important notes or reusable snippets in /memories/ (disk-backed).
- When context grows large, summarize aggressively using the summarization middleware.
- Break down work into clear subtasks; spawn subagents for isolated experiments.
- Be transparent: print tool calls and their results.
"""

# Persistence and workspace settings.
WORKSPACE_ROOT = Path(os.environ.get("CODEX_WORKSPACE_ROOT", os.getcwd()))
PERSIST_DIR = Path(os.environ.get("CODEX_AGENT_HOME", WORKSPACE_ROOT / ".codex_agent"))
CHECKPOINT_PATH = PERSIST_DIR / "checkpoints.sqlite3"
STORE_PATH = PERSIST_DIR / "store.sqlite3"
MEMORY_NAMESPACE = "/memories/"

# Maximum number of parallel subagents (enforced by middleware).
MAX_SUBAGENTS = int(os.environ.get("CODEX_MAX_SUBAGENTS", "3"))

# If set, the first turn is seeded with this text (otherwise CLI will prompt).
DEFAULT_TASK = os.environ.get("CODEX_DEFAULT_TASK", "").strip()

# =========================
# Data classes
# =========================


@dataclass
class MemoryConfig:
    """Configuration for long-context handling."""

    summarization_enabled: bool = True
    summary_model: str = SUMMARY_MODEL
    summary_prompt: str = SUMMARY_PROMPT
    summary_token_limit: int | None = 6000
    # Where to store checkpoints: "memory" or "sqlite"
    checkpoint_backend: str = "sqlite"
    checkpoint_path: Path = CHECKPOINT_PATH
    # Where to store /memories/ content
    store_path: Path = STORE_PATH
    namespace: str = MEMORY_NAMESPACE


@dataclass
class AgentSettings:
    model_name: str = MODEL_NAME
    model_provider: Optional[str] = MODEL_PROVIDER
    model_kwargs: Dict[str, object] = field(default_factory=lambda: dict(MODEL_KWARGS))
    system_prompt: str = SYSTEM_PROMPT
    workspace_root: Path = WORKSPACE_ROOT
    persist_dir: Path = PERSIST_DIR
    max_subagents: int = MAX_SUBAGENTS
    interrupt_on: Dict[str, bool] = field(
        default_factory=lambda: {"write_file": True, "edit_file": True}
    )
    memory: MemoryConfig = field(default_factory=MemoryConfig)


# =========================
# Agent construction
# =========================


def build_backend(settings: AgentSettings):
    """Build a composite backend that routes /memories/ to a persistent directory."""
    from deepagents.backends import CompositeBackend, FilesystemBackend

    mem_root = settings.persist_dir / "memories"
    mem_root.mkdir(parents=True, exist_ok=True)

    fs_backend = FilesystemBackend(root_dir=str(settings.workspace_root), virtual_mode=False)
    mem_backend = FilesystemBackend(root_dir=str(mem_root), virtual_mode=False)

    return CompositeBackend(default=fs_backend, routes={settings.memory.namespace: mem_backend})


def configure_summarization_prompt(settings: AgentSettings):
    """Optionally override the default summarization prompt or disable summarization."""
    try:
        import langchain.agents.middleware.summarization as summary_mod
    except Exception:
        if settings.memory.summarization_enabled:
            print(
                "Warning: summarization middleware not available; skipping summarization.",
                file=sys.stderr,
            )
        settings.memory.summarization_enabled = False
        return

    if settings.memory.summarization_enabled:
        summary_mod.DEFAULT_SUMMARY_PROMPT = settings.memory.summary_prompt
        # If caller set a smaller token limit, update module constant.
        if settings.memory.summary_token_limit is not None:
            summary_mod._DEFAULT_TRIM_TOKEN_LIMIT = settings.memory.summary_token_limit
    else:
        # Replace summarization with a no-op class to effectively disable it.
        class NoopSummary(summary_mod.SummarizationMiddleware):
            def __init__(self, *args, **kwargs):  # noqa: D401
                super().__init__(*args, **kwargs)

            def before_model(self, state, runtime):  # noqa: ARG002
                return None

        summary_mod.SummarizationMiddleware = NoopSummary


def build_checkpointer(memory_cfg: MemoryConfig):
    """Choose between in-memory or sqlite-backed checkpoints."""
    import sqlite3
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver

    if memory_cfg.checkpoint_backend == "sqlite":
        memory_cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(memory_cfg.checkpoint_path, check_same_thread=False)
        return SqliteSaver(conn)
    return MemorySaver()


def build_agent(settings: AgentSettings):
    """Create and compile the deep agent graph."""
    try:
        from langchain.agents import create_deep_agent
    except Exception:
        from deepagents import create_deep_agent
    from langchain.chat_models import init_chat_model

    backend = build_backend(settings)
    checkpointer = build_checkpointer(settings.memory)
    configure_summarization_prompt(settings)

    llm = init_chat_model(
        settings.model_name,
        model_provider=settings.model_provider,
        **settings.model_kwargs,
    )

    agent_graph = create_deep_agent(
        model=llm,
        backend=backend,
        system_prompt=settings.system_prompt,
        checkpointer=checkpointer,
        interrupt_on=settings.interrupt_on,
    )
    return agent_graph.compile() if hasattr(agent_graph, "compile") else agent_graph


# =========================
# CLI + streaming loop
# =========================


class InterruptController:
    """Tracks SIGINT and allows mid-run injection of instructions."""

    def __init__(self):
        self._flag = threading.Event()
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, *_):
        self._flag.set()
        print("\n\nInterrupted. Type new instructions to inject, or 'quit' to exit.\n", flush=True)

    def consume(self) -> bool:
        """Return True if an interrupt was pending and clear it."""
        if self._flag.is_set():
            self._flag.clear()
            return True
        return False


def print_updates(event):
    """Pretty-print assistant/tool messages from a stream update."""
    for node, payload in event.items():
        if payload is None:
            print(f"\n[event::{node}] {payload}", flush=True)
            continue
        messages = payload.get("messages") or []
        if not isinstance(messages, (list, tuple)):
            print(f"\n[event::{node}] {messages}", flush=True)
            continue
        for msg in messages:
            if getattr(msg, "type", None) == "ai":
                print(f"\n[assistant::{node}] {msg.content}", flush=True)
            elif getattr(msg, "type", None) == "tool":
                tool_name = getattr(msg, "name", None) or "tool"
                print(f"\n[tool::{tool_name}] {msg.content}", flush=True)
            elif getattr(msg, "type", None) == "human":
                print(f"\n[user::{node}] {msg.content}", flush=True)


def stream_conversation(app, config, controller: InterruptController, first_message: str):
    """Stream events, handling Ctrl+C to inject instructions mid-run."""
    next_user_msg: Optional[str] = first_message
    while True:
        inputs = {"messages": [("user", next_user_msg)]} if next_user_msg else None
        next_user_msg = None

        try:
            for event in app.stream(inputs, config, stream_mode="updates"):
                if isinstance(event, dict):
                    print_updates(event)
                else:
                    print(f"\n[event] {event}", flush=True)
                if controller.consume():
                    injected = input("Instruction to inject (leave empty to resume): ").strip()
                    if injected.lower() in {"quit", "exit"}:
                        print("Exiting on user request.")
                        return
                    if injected:
                        print(f"[user::inject] {injected}")
                        app.update_state(config, {"messages": [("user", injected)]})
                    print("Resuming...\n")
        except KeyboardInterrupt:
            # Fallback if SIGINT was not caught inside the stream loop.
            injected = input("\nInstruction to inject (leave empty to exit): ").strip()
            if injected:
                app.update_state(config, {"messages": [("user", injected)]})
                continue
            return
        except Exception as exc:  # pragma: no cover - runtime guard
            import traceback

            print(f"Error during streaming: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return

        # After a completed stream turn, prompt for the next instruction.
        user_next = input("\nNext instruction (empty to exit): ").strip()
        if not user_next:
            print("Goodbye.")
            return
        next_user_msg = user_next


# =========================
# Entry point
# =========================


def parse_args():
    parser = argparse.ArgumentParser(
        description="General-purpose long-running coding agent (offline)."
    )
    parser.add_argument(
        "--task",
        help="Seed instruction for the first turn (overrides DEFAULT_TASK).",
    )
    parser.add_argument(
        "--thread-id",
        default="codex-session",
        help="Thread/session identifier for checkpointing.",
    )
    parser.add_argument(
        "--memory-backend",
        choices=["memory", "sqlite"],
        default="sqlite",
        help="Checkpoint backend.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable summarization middleware.",
    )
    parser.add_argument(
        "--workspace",
        default=str(WORKSPACE_ROOT),
        help="Workspace root for filesystem tools.",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(PERSIST_DIR),
        help="Directory for checkpoint/store files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    settings = AgentSettings()
    settings.workspace_root = Path(args.workspace).resolve()
    settings.persist_dir = Path(args.persist_dir).resolve()
    settings.memory.checkpoint_backend = args.memory_backend
    settings.memory.checkpoint_path = settings.persist_dir / CHECKPOINT_PATH.name
    settings.memory.store_path = settings.persist_dir / STORE_PATH.name
    settings.memory.summarization_enabled = not args.no_summary

    # Ensure persistence directory exists before imports that may write files.
    settings.persist_dir.mkdir(parents=True, exist_ok=True)

    # Build app
    try:
        app = build_agent(settings)
    except ImportError as exc:
        print(
            "Missing dependency while building the agent. Install requirements like:\n"
            "  pip install langchain langgraph python-dotenv\n"
            f"ImportError: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    thread_id = args.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    seed = (args.task or DEFAULT_TASK or "").strip()
    if not seed:
        seed = input("Enter the main coding task: ").strip()
        if not seed:
            print("No task provided. Exiting.")
            return

    print(
        f"Workspace: {settings.workspace_root}\n"
        f"Persist dir: {settings.persist_dir}\n"
        f"Checkpointer: {settings.memory.checkpoint_backend}\n"
        f"Summarization: {'on' if settings.memory.summarization_enabled else 'off'}\n"
        "Press Ctrl+C anytime to inject instructions.\n",
        flush=True,
    )

    controller = InterruptController()
    stream_conversation(app, config, controller, seed)


if __name__ == "__main__":
    main()
