# Memory Architecture in Deep Agents

Complete guide to how memory works in the Deep Coding Agent implementation.

## Overview: Three-Layer Memory System

The agent uses **three distinct memory layers**:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Conversation State (Messages)                 │
│  - Current conversation messages                        │
│  - Tool calls and results                               │
│  - Managed by LangGraph state                           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Backend Storage (Files/Data)                  │
│  - File operations (read_file, write_file, etc.)        │
│  - Three backend options: State, Filesystem, Composite  │
│  - Determines persistence strategy                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Checkpointing (Thread State)                  │
│  - Saves entire agent state after each step             │
│  - Enables interrupt/resume                             │
│  - Thread-based conversation history                    │
└─────────────────────────────────────────────────────────┘
```

## Layer 1: Conversation State (Messages)

### What It Is
The conversation history - all user messages, assistant responses, and tool executions.

### How It Works
```python
# Managed by LangGraph's StateGraph
state = {
    "messages": [
        HumanMessage(content="Create a file"),
        AIMessage(content="I'll create that file", tool_calls=[...]),
        ToolMessage(content="File created successfully"),
        AIMessage(content="Done! File created.")
    ]
}
```

### Automatic Summarization
When messages exceed ~170,000 tokens:

```python
# SummarizationMiddleware automatically triggered
# Old messages get summarized:
state["messages"] = [
    AIMessage(content="[SUMMARY] Previously: Created 3 files..."),
    # Recent messages kept in full
    HumanMessage(content="Now add error handling"),
    ...
]
```

### In Your Code
You don't manage this directly - it happens automatically. But you can see it:

```
You: Create multiple files
[Agent creates files...]

You: What did we do so far?
[Agent recalls from message history]

[After 170k tokens, old messages auto-summarize]
You: What did we do at the start?
[Agent recalls from summary, not full messages]
```

## Layer 2: Backend Storage (The Key Part)

This is where **your implementation** offers three options.

### Backend Option 1: StateBackend (Ephemeral)

**How It Works:**
```python
# In claude-agent.py
backend = lambda rt: StateBackend(rt)
```

Files are stored **in LangGraph's conversation state**:

```python
# When agent uses write_file tool:
state["files"] = {
    "/hello.py": "print('hello')",
    "/config.json": "{ ... }"
}
```

**Characteristics:**
- ✅ Fast (in-memory)
- ✅ Simple
- ❌ Lost when agent terminates
- ❌ Files only exist in conversation state
- ❌ Can't access with normal file system tools

**When to Use:**
- Quick experiments
- Testing
- Throwaway tasks

**Example:**
```python
MEMORY_BACKEND = "state"

# Run agent
You: Create hello.py
[Agent creates it in state]

# Exit and restart
python3 claude-agent.py
You: Read hello.py
[File is gone - it was only in state]
```

### Backend Option 2: FilesystemBackend (Persistent)

**How It Works:**
```python
# In claude-agent.py
backend = FilesystemBackend(root_dir=".", virtual_mode=True)
```

Files are **written to actual disk**:

```python
# When agent uses write_file tool:
# Creates actual file on filesystem
with open("./hello.py", "w") as f:
    f.write("print('hello')")
```

**Characteristics:**
- ✅ Persistent across runs
- ✅ Real files you can edit/run
- ✅ Works with git, IDE, etc.
- ✅ Supports execute tool (shell commands)
- ⚠️ Slower than StateBackend
- ⚠️ Files remain after agent exits

**When to Use:**
- Real projects
- Long-running development
- When you need to run/edit files outside agent

**Example:**
```python
MEMORY_BACKEND = "filesystem"
WORKSPACE_DIR = "./my_project"

# Run agent
You: Create hello.py
[Agent creates ./my_project/hello.py - real file!]

# Exit and restart
python3 claude-agent.py
You: Read hello.py
[File still there - it's on disk]

# You can also edit it yourself
vim ./my_project/hello.py
```

### Backend Option 3: CompositeBackend (Hybrid) **[Recommended]**

**How It Works:**
```python
# In claude-agent.py
backend = lambda rt: CompositeBackend(
    default=StateBackend(rt),           # Most files: ephemeral
    routes={
        "/memories/": StoreBackend(rt)   # /memories/: persistent
    }
)
```

**Routes different paths to different backends:**

```
File Path                 → Backend         → Storage
─────────────────────────────────────────────────────────
/temp.py                  → StateBackend    → Ephemeral
/working_file.py          → StateBackend    → Ephemeral
/memories/decisions.md    → StoreBackend    → Persistent
/memories/architecture.md → StoreBackend    → Persistent
```

**Characteristics:**
- ✅ Best of both worlds
- ✅ Fast for temporary files
- ✅ Persistent for important context
- ✅ `/memories/` survives across sessions
- ✅ Recommended for most use cases

**When to Use:**
- Default choice
- Long conversations with context
- When you want some things to persist

**Example:**
```python
MEMORY_BACKEND = "composite"

# Run agent
You: Store in /memories/ that we're using PostgreSQL
[Agent creates /memories/database.md - PERSISTENT]

You: Create temp_analysis.py
[Agent creates it in state - EPHEMERAL]

# Exit and restart
python3 claude-agent.py
You: What database are we using?
[Agent reads /memories/database.md - still there!]

You: Read temp_analysis.py
[File is gone - it was ephemeral]
```

## Layer 3: Checkpointing (Thread State)

### What It Is
Saves the **entire agent state** after each step, enabling:
- Resume conversations
- Interrupt/resume (Ctrl+C)
- Conversation history across sessions

### How It Works
```python
# In claude-agent.py
checkpointer = MemorySaver()  # In-memory checkpoint storage

agent = create_deep_agent(
    checkpointer=checkpointer,
    # ...
)

# Each invocation uses a thread_id
agent.invoke(
    {"messages": [...]},
    {"configurable": {"thread_id": "coding-session-1"}}
)
```

### What Gets Saved
```python
checkpoint = {
    "messages": [...],           # All conversation messages
    "files": {...},              # Files from StateBackend
    "todos": [...],              # Current todos
    "subagent_calls": [...],     # Subagent history
    "tool_results": {...},       # Recent tool outputs
}
```

### Thread IDs
Each conversation has a thread ID:

```python
THREAD_ID = "coding-session-1"

# Same thread = continue conversation
agent.invoke(..., {"configurable": {"thread_id": "coding-session-1"}})
# Remembers previous messages

# New thread = fresh start
agent.invoke(..., {"configurable": {"thread_id": "coding-session-2"}})
# No memory of previous conversation
```

### In Your Code
```python
# First session
You: Create a module for authentication
[Agent creates files, discusses approach]
# Exit

# Second session (same THREAD_ID)
python3 claude-agent.py
You: Continue where we left off
[Agent remembers the authentication module work]

# Or type 'reset' to get new thread
You: reset
[New thread ID generated, fresh start]
```

## The Runtime Lambda Pattern (Important!)

### Why Lambdas?

Some backends need access to **runtime configuration** that's only available when the agent runs:

```python
# WRONG - StateBackend needs runtime
backend = StateBackend()  # Error! Missing runtime argument

# CORRECT - Provide runtime via lambda
backend = lambda rt: StateBackend(rt)
#               ^^
#               Runtime provided by create_deep_agent
```

### What's in Runtime?
```python
class ToolRuntime:
    state: dict         # Current agent state
    context: dict       # Configuration context
    store: BaseStore    # Storage backend
    writer: StreamWriter  # For streaming output
```

### Which Backends Need Lambda?

```python
# Need lambda (require runtime):
StateBackend:      lambda rt: StateBackend(rt)
StoreBackend:      lambda rt: StoreBackend(rt)
CompositeBackend:  lambda rt: CompositeBackend(...)

# Don't need lambda:
FilesystemBackend: FilesystemBackend(root_dir=".")
```

### Your Implementation
```python
def create_memory_backend(config: AgentConfig):
    if config.MEMORY_BACKEND == "state":
        # Returns lambda - StateBackend needs runtime
        return lambda rt: StateBackend(rt)

    elif config.MEMORY_BACKEND == "filesystem":
        # Returns instance - FilesystemBackend doesn't need runtime
        return FilesystemBackend(root_dir=str(workspace_path))

    elif config.MEMORY_BACKEND == "composite":
        # Returns lambda - contains StateBackend and StoreBackend
        return lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={"/memories/": StoreBackend(rt)}
        )
```

## Complete Memory Flow Example

Let's trace a complete interaction:

### Setup
```python
# claude-agent.py config
MEMORY_BACKEND = "composite"
THREAD_ID = "session-123"
ENABLE_CHECKPOINTING = True
```

### Interaction Flow

**Step 1: User Request**
```
You: Create a config file and store our architecture decisions in /memories/
```

**Step 2: Message Added to State**
```python
state["messages"].append(
    HumanMessage(content="Create a config file and store...")
)
```

**Step 3: Agent Plans**
```python
# Agent uses write_todos tool
state["todos"] = [
    "Create config file",
    "Document architecture in /memories/"
]
```

**Step 4: Agent Creates Config**
```python
# Agent uses write_file tool with path "/config.json"
# CompositeBackend routes to StateBackend (default)
# File stored in state (ephemeral)

state["files"]["/config.json"] = '{"db": "postgres"}'
```

**Step 5: Agent Stores Architecture**
```python
# Agent uses write_file tool with path "/memories/architecture.md"
# CompositeBackend routes to StoreBackend (special route)
# File stored in persistent store

store.put("/memories/architecture.md", "We use PostgreSQL...")
```

**Step 6: Checkpoint Saved**
```python
# After agent completes
checkpointer.save({
    "thread_id": "session-123",
    "messages": [...],  # Including all above
    "files": {"/config.json": ...},  # Ephemeral file
    "todos": [...],
})
# Note: /memories/ files already in persistent store
```

**Step 7: User Exits**
```
You: exit
```

### Later Session

**Step 8: Restart Agent**
```bash
python3 claude-agent.py
# Same THREAD_ID = "session-123"
```

**Step 9: State Restored from Checkpoint**
```python
state = checkpointer.load("session-123")
# state["messages"] = previous conversation
# state["files"] = {"/config.json": ...}  # Ephemeral file still there
```

**Step 10: User Continues**
```
You: What architecture decisions did we make?
```

**Step 11: Agent Reads from /memories/**
```python
# Agent uses read_file tool on "/memories/architecture.md"
# CompositeBackend routes to StoreBackend
# Reads from persistent store

content = store.get("/memories/architecture.md")
# Returns: "We use PostgreSQL..."
```

## Memory Management Strategies

### Strategy 1: Short Tasks (State)
```python
MEMORY_BACKEND = "state"
ENABLE_CHECKPOINTING = False

# Use for:
# - Quick one-off tasks
# - Experiments
# - No persistence needed
```

### Strategy 2: Long Development (Filesystem)
```python
MEMORY_BACKEND = "filesystem"
WORKSPACE_DIR = "./my_project"
ENABLE_CHECKPOINTING = True
THREAD_ID = "my-project-dev"

# Use for:
# - Real projects
# - Need to run files outside agent
# - Git integration
```

### Strategy 3: Long Conversations (Composite)
```python
MEMORY_BACKEND = "composite"
ENABLE_CHECKPOINTING = True
ENABLE_SUMMARIZATION = True

# Use for:
# - Extended coding sessions
# - Need context across sessions
# - Balance speed and persistence

# Pattern:
You: Store in /memories/requirements.md our project requirements
[Persists across sessions]

You: Create temp_script.py for analysis
[Ephemeral, fast]
```

## The /memories/ Pattern

### Recommended Usage

**Store in /memories/:**
- Architecture decisions
- Design patterns being used
- API endpoints and contracts
- Database schema
- Important context for future sessions

**Don't store in /memories/:**
- Temporary analysis
- Draft code
- Intermediate calculations

### Example
```
You: We're building a REST API. Store in /memories/:
     - Framework: FastAPI
     - Database: PostgreSQL with SQLAlchemy
     - Auth: JWT tokens
     - Testing: pytest with fixtures

[Later, days later with same THREAD_ID]

You: Add authentication to the API

A:
I'll add JWT authentication to the FastAPI app.
Let me check our architecture decisions...

[Tool Calls:]
  - read_file
    Args: {'path': '/memories/architecture.md'}

Based on our architecture, I'll implement JWT tokens...
```

## Summarization in Detail

### When It Triggers
```python
# Configured in AgentConfig
SUMMARIZATION_THRESHOLD = 170000  # tokens

# Automatically triggered when:
total_tokens = count_tokens(state["messages"])
if total_tokens > SUMMARIZATION_THRESHOLD:
    trigger_summarization()
```

### What Happens
```python
# Before summarization (180k tokens)
messages = [
    HumanMessage("Create module A"),  # 2k tokens
    AIMessage("I'll create..."),      # 5k tokens
    ToolMessage("Created file..."),   # 1k tokens
    # ... 100 more messages ...      # 172k tokens total
    HumanMessage("Now add feature X"),
]

# After summarization
messages = [
    AIMessage("""[SUMMARY of previous conversation]
    Created module A with features X, Y, Z.
    Refactored database layer to use async.
    Added comprehensive tests.
    Outstanding: Feature X implementation.
    """),  # 2k tokens
    # Recent messages kept
    HumanMessage("Now add feature X"),
]
```

### Custom Summarization Prompt
```python
# In AgentConfig
CUSTOM_SUMMARIZATION_PROMPT = """
Summarize the conversation focusing on:

1. **Code Changes**: Files created, modified, deleted
2. **Design Decisions**: Architecture choices and reasoning
3. **Outstanding Tasks**: What's not yet complete
4. **Important Context**: Critical info for continuing

Format as structured markdown.
"""
```

## Debugging Memory Issues

### Check What's in Memory

**1. Check Messages**
```python
# Enable debug mode
DEBUG = True

# You'll see state dumps showing:
# - All messages
# - Files in state
# - Current todos
```

**2. Check Files**
```
You: List all files in the current workspace
[Agent uses ls tool]

You: What's in /memories/?
[Agent reads /memories/ directory]
```

**3. Check Thread State**
```python
# Change THREAD_ID to see if issue is thread-specific
THREAD_ID = "debug-session"
```

### Common Issues

**Issue 1: "Agent forgot previous conversation"**
```python
# Check: Are you using same THREAD_ID?
THREAD_ID = "coding-session-1"  # Must be same across runs

# Check: Is checkpointing enabled?
ENABLE_CHECKPOINTING = True
```

**Issue 2: "Files disappeared"**
```python
# Check: What backend are you using?
MEMORY_BACKEND = "state"  # Files disappear on exit

# Fix: Use filesystem or composite
MEMORY_BACKEND = "composite"
# Store important files in /memories/
```

**Issue 3: "Agent can't remember early conversation"**
```python
# Likely: Summarization happened (>170k tokens)

# Fix: Important context should be in /memories/
You: Store the key decisions in /memories/decisions.md
```

## Memory Performance Tips

### Tip 1: Use Appropriate Backend
```python
# Fast, disposable
MEMORY_BACKEND = "state"

# Persistent, real files
MEMORY_BACKEND = "filesystem"

# Best balance (recommended)
MEMORY_BACKEND = "composite"
```

### Tip 2: Manage Summarization
```python
# More aggressive (summarize sooner)
SUMMARIZATION_THRESHOLD = 100000

# Less aggressive (keep more history)
SUMMARIZATION_THRESHOLD = 200000
```

### Tip 3: Use /memories/ Wisely
```python
# Good: Store persistent context
You: Store our API design in /memories/api.md

# Bad: Store temporary data
You: Store this debug output in /memories/  # Don't do this
```

### Tip 4: Thread Management
```python
# Long project: One thread
THREAD_ID = "project-alpha"

# Different tasks: Different threads
THREAD_ID = "feature-auth"
THREAD_ID = "feature-api"
THREAD_ID = "bugfix-123"
```

## Summary

| Aspect | StateBackend | FilesystemBackend | CompositeBackend |
|--------|-------------|-------------------|------------------|
| **Speed** | Fast | Slower | Fast (mostly) |
| **Persistence** | No | Yes | Selective |
| **Files** | In state | On disk | Both |
| **Use Case** | Experiments | Real projects | Most use cases |
| **Best For** | Quick tasks | Development | Long sessions |

**Key Concepts:**
1. **Three layers**: Messages → Backend → Checkpoints
2. **Backends determine file storage**: State (ephemeral), Filesystem (persistent), Composite (hybrid)
3. **Checkpointing enables resume**: Same thread = continue conversation
4. **Summarization handles long context**: Auto-compresses >170k tokens
5. **/memories/ for persistence**: Store important context here with CompositeBackend
6. **Lambda pattern for runtime**: Some backends need runtime access

**Recommended Setup:**
```python
MEMORY_BACKEND = "composite"          # Hybrid storage
ENABLE_CHECKPOINTING = True           # Resume support
ENABLE_SUMMARIZATION = True           # Handle long conversations
THREAD_ID = "my-project"              # Consistent thread
```

This gives you the best balance of speed, persistence, and memory management!
