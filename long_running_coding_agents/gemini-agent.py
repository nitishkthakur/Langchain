import os
import sys
import time
import signal
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import Tool, BaseTool
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.tools.file_management.list_dir import ListDirectoryTool
import subprocess
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

@tool
def run_shell_command(command: str) -> str:
    """Run shell commands. Use this to execute python scripts like 'python3 main.py' or 'mkdir'.
    
    Args:
        command: The command to run.
    """
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        output = result.stdout
        if result.stderr:
            output += f"\nStderr: {result.stderr}"
        return output
    except Exception as e:
        return f"Error executing command: {e}"

# --- Configuration ---
TASK = """
1. Create a directory named 'my_project_v2'.
2. Inside 'my_project_v2', create a file named 'hello.py' with content 'print("Hello form Gemini CLI")'.
3. Run 'hello.py' and show me the output.
"""

MODEL_NAME = "claude-3-haiku-20240307"
MAX_HISTORY_MESSAGES = 20 # Keep last N messages to simulate limited context window

# --- End of Configuration ---

load_dotenv()

# --- Mock LLM ---
class MockChatModel(BaseChatModel):
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        last_msg_content = messages[-1].content if isinstance(messages[-1].content, str) else ""
        
        # Simple Mock Logic
        if "Create a directory" in last_msg_content or "mkdir" in last_msg_content.lower():
            # Simulate tool call
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="",
                tool_calls=[{
                    "name": "run_shell_command", 
                    "args": {"command": "mkdir -p my_project_v2"}, 
                    "id": "call_1"
                }]
            ))])
        elif "create a file" in last_msg_content.lower():
             return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="",
                tool_calls=[{
                    "name": "write_file", 
                    "args": {"file_path": "my_project_v2/hello.py", "content": 'print("Hello from Gemini CLI")'},
                    "id": "call_2"
                }]
            ))])
        elif "run" in last_msg_content.lower() and "hello.py" in last_msg_content.lower():
             return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="",
                tool_calls=[{
                    "name": "run_shell_command", 
                    "args": {"command": "python3 my_project_v2/hello.py"}, 
                    "id": "call_3"
                }]
            ))])
        
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="[MOCK] Task completed. I have executed the necessary commands."))])

    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def bind_tools(self, tools, **kwargs):
        # Mock bind_tools just returns self, assuming it knows about tools implicitly
        return self

# --- Signal Handler ---
interrupted = False
def signal_handler(sig, frame):
    global interrupted
    print("\n\n⚠️ User interruption (Ctrl+C). Pausing...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

# --- Initialization ---
def init_chat_model() -> BaseChatModel:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print(f"🔌 Connected to Anthropic ({MODEL_NAME})")
        return ChatAnthropic(model=MODEL_NAME, temperature=0, api_key=api_key)
    
    print("⚠️  No API Key found. Using Mock Model for demonstration.")
    return MockChatModel()

def main():
    global interrupted
    print("🚀 Initializing Gemini CLI Agent (Deep Coding Agent)...")
    
    # Tools
    tools = [
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool(),
        run_shell_command
    ]
    
    # Rename ShellTool to 'run_shell_command' to match what models might prefer or my mock expects
    # Actually ShellTool name is 'terminal' usually. Let's check.
    # We can wrap it or just use it.
    
    llm = init_chat_model()
    
    # Create the agent graph
    system_prompt = "You are an expert software engineer. You have access to the filesystem. Execute the user's requests step by step."
    
    try:
        agent_graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            # debug=True 
        )
    except Exception as e:
        print(f"❌ Failed to create agent graph: {e}")
        return

    # Chat History (State Management)
    # We start with empty history. The 'input' to the graph is the list of messages.
    # The graph returns the *final* list of messages (or state).
    # We will maintain our own history and append to it.
    
    chat_history: List[BaseMessage] = []
    
    # Initial Task
    next_input = TASK
    
    while True:
        if next_input:
            print(f"\n📝 Task: {next_input.strip()[:50]}...")
            
            # Prepare input messages
            # If we have history, we pass it. If not, we start.
            # Actually, `create_agent` graph expects 'messages' key.
            # We append the new user message to our history.
            
            chat_history.append(HumanMessage(content=next_input))
            
            # Memory Management: Summarize or Truncate if too long
            # (Simple truncation for this demo)
            if len(chat_history) > MAX_HISTORY_MESSAGES:
                print("🧹 Compressing memory (keeping last N messages)...")
                # Keep system message if we had one (create_agent handles system prompt internally usually)
                # We just keep the last N.
                chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
            
            try:
                # Invoke the agent
                # The agent might print to stdout if we configured callbacks, but `create_agent` 
                # might not use the same callback system as AgentExecutor.
                # We can stream the output to show progress.
                
                print("\n🤖 Agent Working...")
                
                final_state = None
                # We iterate/stream to show activity
                for event in agent_graph.stream({"messages": chat_history}, stream_mode="updates"):
                    # Event is a dict of node_name -> state_update
                    for node, update in event.items():
                        if "messages" in update:
                            new_msgs = update["messages"]
                            for msg in new_msgs:
                                if isinstance(msg, AIMessage):
                                    if msg.tool_calls:
                                        for tc in msg.tool_calls:
                                            print(f"  🛠️  Call: {tc['name']}({tc['args']})")
                                    elif msg.content:
                                        print(f"  🗣️  Assistant: {msg.content}")
                                elif isinstance(msg, ToolMessage):
                                    print(f"  ↳ Result: {msg.content[:100]}...")
                
                # Update our history with the result
                # We need to run invoke to get full state or just assume stream gave us everything.
                # Safest is to get final state or just append what we got.
                # create_agent state usually replaces or appends.
                # Let's assume we need to update `chat_history` with the NEW messages.
                
                # A better way with `create_agent` is to pass the whole history and let it return the whole history (or diff).
                # `create_agent` returns a compiled graph.
                # invoke returns the final state.
                
                final_state = agent_graph.invoke({"messages": chat_history})
                chat_history = final_state["messages"]
                
                # Print final response if not already printed
                last_msg = chat_history[-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    print(f"\n✅ Final Answer: {last_msg.content}")

            except Exception as e:
                print(f"❌ Execution Error: {e}")
                # import traceback
                # traceback.print_exc()
            
            next_input = None # Reset
        
        # Interactive Loop
        try:
            if interrupted:
                print("\npaused. Type 'resume' or new instructions, or 'exit'.")
                interrupted = False
            
            user_input = input("\n(agent idle) > ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            elif user_input.strip():
                next_input = user_input
                
        except KeyboardInterrupt:
            interrupted = True
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()