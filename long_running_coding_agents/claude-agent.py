#!/usr/bin/env python3
"""
General-purpose coding agent for long-running tasks using LangChain's create_deep_agent.

This agent is designed to handle complex, multi-step coding tasks with extensive context
requirements. It includes memory management, checkpointing, and interrupt/resume support.

Features:
- Multiple memory backend options (StateBackend, FilesystemBackend, CompositeBackend)
- Automatic summarization for long contexts
- CLI interface with streaming output
- Ctrl+C interrupt handling with resume capability
- Checkpoint-based state persistence
- Filesystem tools for code manipulation
- Subagent spawning for complex tasks
"""

import os
import sys
import signal
import asyncio
from typing import Literal, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents import create_deep_agent
from deepagents.backends import (
    StateBackend,
    FilesystemBackend,
    CompositeBackend,
    StoreBackend,
)


# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================

class AgentConfig:
    """Configuration for the coding agent."""

    # Model configuration
    # Use init_chat_model format: "provider:model-name"
    # Examples: "anthropic:claude-sonnet-4-5-20250929", "openai:gpt-4o", "openai:gpt-4o-mini"
    MODEL = "anthropic:claude-3-5-haiku-20241022"

    # Memory backend type
    # - "state": Ephemeral in-memory (no persistence between runs)
    # - "filesystem": Persistent filesystem storage
    # - "composite": Hybrid - default ephemeral + /memories/ persisted
    MEMORY_BACKEND: Literal["state", "filesystem", "composite"] = "filesystem"

    # Filesystem configuration (used for "filesystem" and "composite" backends)
    WORKSPACE_DIR = "./agent_workspace"
    MEMORIES_DIR = "/memories/"  # Special directory for persistent memories

    # Summarization configuration
    # Enable automatic summarization for long conversations
    ENABLE_SUMMARIZATION = True
    SUMMARIZATION_THRESHOLD = 170000  # Tokens before summarization kicks in

    # Custom summarization prompt (optional)
    # If None, uses default deepagents summarization
    CUSTOM_SUMMARIZATION_PROMPT: Optional[str] = None
    # Example custom prompt:
    # """Summarize the conversation focusing on:
    # 1. Key decisions made
    # 2. Code changes implemented
    # 3. Outstanding tasks
    # 4. Important context for continuing the work
    # """

    # Checkpointing for interrupt/resume
    ENABLE_CHECKPOINTING = True

    # Thread ID for conversation continuity
    # Change this to start a new conversation
    THREAD_ID = "coding-session-1"

    # System prompt for the agent
    SYSTEM_PROMPT = """You are an expert coding assistant specialized in long-running,
complex software development tasks. You have access to filesystem tools, can plan tasks
using todos, and can spawn subagents for specialized work.

Guidelines:
- Break down complex tasks into manageable steps using the write_todos tool
- Use filesystem tools to read, write, and edit code files
- When encountering large codebases, use glob and grep to navigate efficiently
- For specialized tasks, consider spawning subagents with the task tool
- Always test your changes when possible
- Provide clear explanations of your changes and reasoning
- Handle errors gracefully and suggest fixes

When working on long-running tasks:
1. Start by understanding the codebase structure
2. Plan your approach using todos
3. Implement changes incrementally
4. Test and verify each change
5. Document important decisions in /memories/ for future reference
"""

    # Interrupt configuration
    # Tools that require human approval before execution
    INTERRUPT_ON = {
        # "write_file": True,  # Uncomment to require approval for file writes
        # "execute": True,     # Uncomment to require approval for shell commands
    }

    # Debug mode - shows internal state and detailed logging
    DEBUG = False

    # Additional custom tools (optional)
    # Add your custom tools here as a list
    CUSTOM_TOOLS = []


# ============================================================================
# Agent Setup
# ============================================================================

def create_memory_backend(config: AgentConfig):
    """Create the appropriate memory backend based on configuration.

    Note: StateBackend and StoreBackend require runtime, so they must be
    returned as lambdas. FilesystemBackend can be returned directly.
    """
    if config.MEMORY_BACKEND == "state":
        print("Using StateBackend (ephemeral in-memory storage)")
        # StateBackend requires runtime parameter, return as lambda
        return lambda rt: StateBackend(rt)

    elif config.MEMORY_BACKEND == "filesystem":
        workspace_path = Path(config.WORKSPACE_DIR).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        print(f"Using FilesystemBackend (persistent storage at {workspace_path})")
        # FilesystemBackend can be instantiated directly
        return FilesystemBackend(root_dir=".", virtual_mode=True)

    elif config.MEMORY_BACKEND == "composite":
        workspace_path = Path(config.WORKSPACE_DIR).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        print(f"Using CompositeBackend (ephemeral + persistent /memories/)")
        print(f"  - Default: ephemeral in-memory")
        print(f"  - {config.MEMORIES_DIR}: persistent storage")

        # CompositeBackend with StateBackend requires runtime, return as lambda
        return lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={
                config.MEMORIES_DIR: StoreBackend(rt)
            },
        )

    else:
        raise ValueError(f"Unknown memory backend: {config.MEMORY_BACKEND}")


def create_coding_agent(config: AgentConfig):
    """Create the deep coding agent with all configurations."""

    # Initialize the chat model
    print(f"\nInitializing model: {config.MODEL}")
    model = init_chat_model(config.MODEL)

    # Create memory backend
    backend = create_memory_backend(config)

    # Create checkpointer if enabled
    checkpointer = MemorySaver() if config.ENABLE_CHECKPOINTING else None
    if config.ENABLE_CHECKPOINTING:
        print("Checkpointing enabled for interrupt/resume support")

    # Create store for StoreBackend (needed for composite backend)
    store = InMemoryStore() if config.MEMORY_BACKEND == "composite" else None

    # Create the agent
    print("\nCreating deep agent with middleware:")
    print("  - TodoListMiddleware (task planning)")
    print("  - FilesystemMiddleware (file operations)")
    print("  - SubAgentMiddleware (spawn specialized agents)")
    if config.ENABLE_SUMMARIZATION:
        print(f"  - SummarizationMiddleware (auto-summarize at {config.SUMMARIZATION_THRESHOLD} tokens)")

    agent = create_deep_agent(
        model=model,
        tools=config.CUSTOM_TOOLS,
        system_prompt=config.SYSTEM_PROMPT,
        backend=backend,
        interrupt_on=config.INTERRUPT_ON,
        checkpointer=checkpointer,
        store=store,
        debug=config.DEBUG,
    )

    print("\nAgent ready!")
    return agent


# ============================================================================
# CLI Interface
# ============================================================================

class InterruptHandler:
    """Handle Ctrl+C interrupts gracefully."""

    def __init__(self):
        self.interrupted = False
        self.original_handler = None

    def __enter__(self):
        self.interrupted = False
        self.original_handler = signal.signal(signal.SIGINT, self._handle_interrupt)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)

    def _handle_interrupt(self, signum, frame):
        self.interrupted = True
        print("\n\n[Interrupt detected - agent will pause after current operation]")
        print("[Press Ctrl+C again to force quit]")
        # Restore default handler for second Ctrl+C
        signal.signal(signal.SIGINT, signal.default_int_handler)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_message(message):
    """Pretty print a message."""
    if isinstance(message, HumanMessage):
        print("\n" + "─" * 80)
        print("USER:")
        print("─" * 80)
        print(message.content)

    elif isinstance(message, AIMessage):
        print("\n" + "─" * 80)
        print("ASSISTANT:")
        print("─" * 80)

        # Print content
        if message.content:
            print(message.content)

        # Print tool calls if any
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print("\n[Tool Calls:]")
            for tool_call in message.tool_calls:
                print(f"  - {tool_call['name']}")
                if 'args' in tool_call:
                    # Print args in a compact way
                    args_str = str(tool_call['args'])
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    print(f"    Args: {args_str}")

    elif isinstance(message, ToolMessage):
        print(f"\n[Tool Result: {message.name}]")
        content = message.content
        if len(str(content)) > 500:
            content = str(content)[:500] + "\n... (truncated)"
        print(content)


async def run_agent_stream(agent, user_input: str, config: AgentConfig):
    """Run the agent with streaming output and interrupt handling."""

    messages = [HumanMessage(content=user_input)]

    # Configuration for invoke/stream
    run_config = {
        "configurable": {
            "thread_id": config.THREAD_ID,
        }
    }

    print_separator()
    print(f"Processing (thread: {config.THREAD_ID})...")
    print_separator()

    # Use interrupt handler
    with InterruptHandler() as handler:
        try:
            # Stream the agent's response
            async for chunk in agent.astream(
                {"messages": messages},
                run_config,
                stream_mode="values"
            ):
                # Check for interrupt
                if handler.interrupted:
                    print("\n[Pausing agent execution...]")
                    return "interrupted"

                # Print the latest message
                if "messages" in chunk and chunk["messages"]:
                    latest_message = chunk["messages"][-1]

                    # Only print if it's a new message (not already printed)
                    # This is a simple approach - in production you'd want better tracking
                    print_message(latest_message)

            return "completed"

        except KeyboardInterrupt:
            print("\n[Force quit detected]")
            return "force_quit"

        except Exception as e:
            print(f"\n[Error: {e}]")
            if config.DEBUG:
                import traceback
                traceback.print_exc()
            return "error"


async def interactive_loop(agent, config: AgentConfig):
    """Run interactive CLI loop with interrupt/resume support."""

    print_separator("=")
    print("DEEP CODING AGENT")
    print_separator("=")
    print(f"\nThread ID: {config.THREAD_ID}")
    print("Commands:")
    print("  - Type your message and press Enter")
    print("  - Press Ctrl+C to interrupt and add new instructions")
    print("  - Type 'exit' or 'quit' to end the session")
    print("  - Type 'reset' to start a new conversation")
    print_separator("=")

    while True:
        try:
            # Get user input
            print("\n")
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            if user_input.lower() == "reset":
                # Generate new thread ID
                import time
                config.THREAD_ID = f"coding-session-{int(time.time())}"
                print(f"\n[New conversation started: {config.THREAD_ID}]")
                continue

            # Run the agent
            status = await run_agent_stream(agent, user_input, config)

            # Handle different statuses
            if status == "interrupted":
                print("\n[Agent paused. You can now add new instructions.]")
                print("[The conversation will continue from where it left off.]")
                continue

            elif status == "force_quit":
                print("\n[Session terminated]")
                break

            elif status == "error":
                print("\n[An error occurred. You can continue or type 'exit' to quit.]")
                continue

        except EOFError:
            print("\n\nGoodbye!")
            break

        except Exception as e:
            print(f"\n[Unexpected error: {e}]")
            if config.DEBUG:
                import traceback
                traceback.print_exc()


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Create configuration
    config = AgentConfig()

    # Create agent
    agent = create_coding_agent(config)

    # Run interactive loop
    await interactive_loop(agent, config)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
