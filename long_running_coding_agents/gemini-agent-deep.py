import os
import sys
import subprocess
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

# Ensure we can import from the environment if not already in path (though venv activation handles this)
# import deepagents # Check import

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deepagents.graph import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from langgraph.checkpoint.sqlite import SqliteSaver

# --- Configuration ---
TASK = """
1. Create a directory named 'gemini_deep_project'.
2. Inside it, create a file 'factorial.py' that calculates factorial of a number provided as arg.
3. Run it with argument 5 and print the result.
4. If successful, write a file 'success.txt' with the result.
"""

MODEL_NAME = "claude-3-haiku-20240307"
MEMORY_DB_PATH = ".gemini_agent_memory.db"
THREAD_ID = "gemini-cli-session"

# --- End of Configuration ---

load_dotenv()

# --- Custom Backend for Local Execution ---
class LocalSandboxBackend(FilesystemBackend):
    """
    Extends FilesystemBackend to add local execution capabilities,
    satisfying SandboxBackendProtocol.
    """
    
    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell command locally."""
        print(f"\n[System] Executing: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes timeout
                cwd=self.cwd
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
                
            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=False
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output="Command timed out after 120 seconds.",
                exit_code=124,
                truncated=False
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Execution error: {str(e)}",
                exit_code=1,
                truncated=False
            )

    @property
    def id(self) -> str:
        return "local_machine"

# --- Signal Handler ---
interrupted = False
def signal_handler(sig, frame):
    global interrupted
    print("\n\n⚠️ User interruption (Ctrl+C). Pausing...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

# --- Initialization ---
def init_chat_model():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  No ANTHROPIC_API_KEY found. Agent might fail if it tries to call LLM.")
        # We could return a mock, but create_deep_agent expects a real model usually.
        # Let's let it fail or assume user will provide it.
        pass
    
    print(f"🔌 Connected to Anthropic ({MODEL_NAME})")
    return ChatAnthropic(model=MODEL_NAME, temperature=0, api_key=api_key)

def main():
    global interrupted
    print("🚀 Initializing Gemini Deep Agent...")

    # 1. Initialize Model
    llm = init_chat_model()

    # 2. Initialize Backend (Filesystem + Execution)
    backend = LocalSandboxBackend(root_dir=Path.cwd(), virtual_mode=False)

    # 3. Initialize Checkpointer (Persistence)
    # We use SqliteSaver to persist the graph state (memory)
    with SqliteSaver.from_conn_string(MEMORY_DB_PATH) as conn:

        # 4. Create Deep Agent
        agent_graph = create_deep_agent(
            model=llm,
            backend=backend,
            checkpointer=conn,
            debug=False
        )

        print("✅ Agent Graph Created.")
        print(f"📂 Working Directory: {backend.cwd}")
        print(f"💾 Memory DB: {MEMORY_DB_PATH}")

        # 5. Execution Loop
        
        config = {"configurable": {"thread_id": THREAD_ID}}
        
        current_state = agent_graph.get_state(config)
        
        next_input = None
        
        if current_state.values:
            print("\n🔄 Resuming previous session...")
            print("Last state found. Type 'status' to see last message, or enter new instruction.")
        else:
            print("\n🆕 Starting new session.")
            next_input = TASK

        while True:
            # Input Handling
            if not next_input:
                try:
                    user_input = input("\n(agent) > ")
                    if user_input.lower() in ['exit', 'quit']:
                        print("Exiting.")
                        break
                    elif user_input.lower() == 'status':
                        current_state = agent_graph.get_state(config)
                        if current_state.values and "messages" in current_state.values:
                            print(f"Last Message: {current_state.values['messages'][-1]}")
                        continue
                    
                    next_input = user_input
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting.")
                    break

            if interrupted:
                interrupted = False
                next_input = None
                continue

            if next_input:
                print(f"\n👤 User: {next_input[:100]}...")
                
                try:
                    inputs = {"messages": [HumanMessage(content=next_input)]}
                    
                    print("🤖 Agent Running...")
                    
                    for event in agent_graph.stream(inputs, config=config, stream_mode="updates"):
                        if interrupted:
                            print("🛑 Interrupting agent execution...")
                            break
                            
                        for node, update in event.items():
                                                                            if "messages" in update:
                                                                                msgs = update["messages"]
                                                                                # Hack for Overwrite object
                                                                                if hasattr(msgs, "value"):
                                                                                    msgs = msgs.value
                                                                                if not isinstance(msgs, list):
                                                                                    msgs = [msgs]
                                                    
                                                                                for msg in msgs:
                                                                                    if isinstance(msg, AIMessage):
                                                                                        if msg.tool_calls:
                                                                                            for tc in msg.tool_calls:
                                                                                                print(f"  🛠️  Tool Call: {tc['name']}")
                                                                                        elif msg.content:
                                                                                            print(f"  🗣️  Assistant: {msg.content}")
                                                                                    elif isinstance(msg, ToolMessage):
                                                                                        content_preview = str(msg.content)[:200]
                                                                                        print(f"  ↳ Tool Output: {content_preview}...")                    
                    if interrupted:
                        print("Paused. State saved.")
                        interrupted = False
                    else:
                        print("✅ Step Completed.")
                        
                except Exception as e:
                    print(f"❌ Error during execution: {e}")
                    import traceback
                    traceback.print_exc()

            next_input = None

if __name__ == "__main__":
    main()
