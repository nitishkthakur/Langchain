"""
LangGraph ReACT Agent with MCP Tools and Middleware

This module implements a ReACT (Reasoning and Acting) Agent using LangGraph's
native graph architecture with nodes, edges, START and END states.

Features:
- Custom ReACTAgent class with configurable model and prompt
- LangGraph nodes and edges (no readymade create_agent)
- MCP server integration (filesystem, search, subagent servers)
- Middleware support (Summarization, TodoList, ShellTool, ToolRetry)
- Checkpointing for conversation persistence
- Async operation support

Author: Generated for workspace
Date: November 2024
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal, Sequence, Union

from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangGraph imports - using native graph construction
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

# MCP imports
from langchain_mcp_adapters import MultiServerMCPClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Helper tools for ReACT agent
from langchain_core.tools import tool

def create_todo_tools() -> List[BaseTool]:
    """Create todo list management tools."""
    todos_storage: List[Dict[str, Any]] = []
    
    @tool
    def write_todos(todos: List[Dict[str, str]]) -> str:
        """Write or update the todo list. Each todo should have 'task' and 'status' keys.
        Status can be: 'pending', 'in_progress', 'done'.
        
        Args:
            todos: List of todo items with 'task' and 'status' fields
        """
        nonlocal todos_storage
        todos_storage = todos
        return f"Updated todo list with {len(todos)} items"
    
    @tool
    def read_todos() -> str:
        """Read the current todo list."""
        if not todos_storage:
            return "No todos found. Use write_todos to create a task list."
        result = "Current TODO List:\n"
        for i, todo in enumerate(todos_storage, 1):
            status = todo.get('status', 'pending')
            task = todo.get('task', 'Unknown task')
            result += f"{i}. [{status.upper()}] {task}\n"
        return result
    
    return [write_todos, read_todos]

def create_shell_tool() -> BaseTool:
    """Create a simple shell execution tool."""
    @tool
    def run_shell_command(command: str, cwd: Optional[str] = None) -> str:
        """Execute a shell command in the workspace.
        
        Args:
            command: The shell command to execute
            cwd: Optional working directory (relative to workspace root)
        """
        import subprocess
        
        workspace = Path("/home/nitish/Documents/github/Langchain")
        working_dir = workspace / cwd if cwd else workspace
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            output = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            return output
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    return run_shell_command


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Workspace configuration
WORKSPACE_ROOT = Path("/home/nitish/Documents/github/Langchain")
TOOLS_DIR = WORKSPACE_ROOT / "tools"

# MCP Server configurations
MCP_SERVER_CONFIGS = {
    "filesystem": {
        "command": "python",
        "args": [str(TOOLS_DIR / "mcp_filesystem.py")],
        "transport": "stdio",
        "env": {"MCP_FS_ROOT": str(WORKSPACE_ROOT)}
    },
    "search": {
        "command": "python", 
        "args": [str(TOOLS_DIR / "mcp_search_server.py")],
        "transport": "stdio"
    },
    "subagent": {
        "command": "python",
        "args": [str(TOOLS_DIR / "mcp_subagent.py")],
        "transport": "stdio",
        "env": {"MCP_WORKSPACE_ROOT": str(WORKSPACE_ROOT)}
    }
}


class AgentState(TypedDict):
    """State schema for the ReACT agent graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class MCPToolManager:
    """Manager for MCP server connections and tools."""
    
    def __init__(self, server_configs: Dict[str, Dict[str, Any]]):
        self.server_configs = server_configs
        self._client: Optional[MultiServerMCPClient] = None
        self._tools: List[BaseTool] = []
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to all configured MCP servers."""
        if self._connected:
            logger.info("Already connected to MCP servers")
            return
        
        try:
            self._client = MultiServerMCPClient(self.server_configs)
            await asyncio.wait_for(self._client.connect(), timeout=30.0)
            self._tools = await self._client.get_tools()
            self._connected = True
            logger.info(f"Connected to MCP servers. Loaded {len(self._tools)} tools.")
        except asyncio.TimeoutError:
            logger.error("Connection to MCP servers timed out")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to MCP servers: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from all MCP servers."""
        if self._client and self._connected:
            await self._client.disconnect()
            self._connected = False
            logger.info("Disconnected from MCP servers")
    
    @property
    def tools(self) -> List[BaseTool]:
        """Get the loaded MCP tools."""
        return self._tools
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to MCP servers."""
        return self._connected


class ReACTAgentBuilder:
class ReACTAgent:
    """
    Custom ReACT (Reasoning and Acting) Agent implementation using LangGraph.
    
    This agent uses LangGraph's native StateGraph with explicit nodes and edges
    instead of relying on readymade create_agent functions.
    
    The agent follows the ReACT pattern:
    1. Call the model to decide next action
    2. Execute tools if requested
    3. Return to model with tool results
    4. Repeat until completion
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 15,
        use_sqlite_checkpoint: bool = False,
        sqlite_path: str = "./checkpoints.db"
    ):
        """
        Initialize the ReACT Agent.
        
        Args:
            model: Model name to use (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            temperature: Model temperature for generation
            system_prompt: System prompt to guide agent behavior
            tools: List of tools available to the agent
            max_iterations: Maximum reasoning/acting iterations
            use_sqlite_checkpoint: Whether to use SQLite for persistence
            sqlite_path: Path to SQLite database for checkpointing
        """
        self.model_name = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.tools = tools or []
        self._iteration_count = 0
        
        # Initialize the language model
        self.llm = init_chat_model(
            model,
            temperature=temperature,
            timeout=60.0,
            max_tokens=4096
        )
        
        # Bind tools to the model
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
        
        # Set up system prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Create checkpointer
        if use_sqlite_checkpoint:
            self.checkpointer = SqliteSaver.from_conn_string(sqlite_path)
            logger.info(f"Using SQLite checkpointer at {sqlite_path}")
        else:
            self.checkpointer = InMemorySaver()
            logger.info("Using in-memory checkpointer")
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"ReACT Agent initialized with model: {model}, {len(self.tools)} tools")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent."""
        return """You are a helpful AI assistant with access to powerful tools.

You have access to:
1. **Filesystem tools** - Read, write, and modify files in the workspace
2. **Search tools** - Search the web and Wikipedia for information
3. **Subagent tools** - Specialized helpers for code review, debugging, testing, etc.
4. **Shell tools** - Execute shell commands in the workspace
5. **Todo list** - Track your progress on complex multi-step tasks

When working on complex tasks:
1. First, create a todo list to plan your approach
2. Work through each task systematically
3. Update the todo list as you make progress
4. Use appropriate tools for each subtask

Be thorough, accurate, and explain your reasoning. When using tools, check the results
and handle any errors gracefully.

IMPORTANT: When you have completed the task or provided a final answer, do NOT call any more tools."""
    
    def _build_graph(self):
        """Build the LangGraph StateGraph with nodes and edges."""
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._execute_tools)
        
        # Add edges
        # START -> agent (entry point)
        workflow.add_edge(START, "agent")
        
        # agent -> tools OR agent -> END (conditional)
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # tools -> agent (loop back for next iteration)
        workflow.add_edge("tools", "agent")
        
        # Compile the graph with checkpointer
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("ReACT agent graph built: START -> agent <-> tools -> END")
        return compiled_graph
    
    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """
        Node: Call the language model to get next action.
        
        This is the reasoning step in ReACT.
        """
        messages = state["messages"]
        
        # Add system prompt if this is the first call
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system_prompt)] + list(messages)
        
        # Call the model
        response = self.llm.invoke(messages)
        
        # Increment iteration counter
        self._iteration_count += 1
        
        return {"messages": [response]}
    
    def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        """
        Node: Execute the tools requested by the model.
        
        This is the acting step in ReACT.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # Create the tool node for execution
        tool_node = ToolNode(self.tools)
        
        # Execute tools
        result = tool_node.invoke(state)
        
        return result
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """
        Conditional edge: Decide whether to continue to tools or end.
        
        Returns:
            "continue" if tools should be called
            "end" if agent is done
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check iteration limit
        if self._iteration_count >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached")
            return "end"
        
        # If the last message has tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        # Otherwise, end
        return "end"
    
    def add_tools(self, tools: List[BaseTool]) -> None:
class ReACTAgentRunner:
    """Runner for executing the ReACT agent with conversation management."""
    
    def __init__(
        self,
        agent: ReACTAgent,
        mcp_manager: Optional[MCPToolManager] = None
    ):
        self.agent = agent
        self.mcp_manager = mcp_manager
        self._thread_id = 1
    async def ainvoke(
        self,
        messages: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously invoke the agent.
        
        Args:
            messages: Input message(s) - can be a string, list of messages, or state dict
            config: Configuration including thread_id for checkpointing
            
        Returns:
            Final state with all messages
        """
        # Reset iteration counter for each invocation
        self._iteration_count = 0
        
        # Normalize input to state dict
        if isinstance(messages, str):
            input_state = {"messages": [HumanMessage(content=messages)]}
        elif isinstance(messages, list):
            input_state = {"messages": messages}
        else:
            input_state = messages
        
        # Invoke the graph
        result = await self.graph.ainvoke(input_state, config=config)
        return result
    
    def invoke(
        self,
        messages: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronously invoke the agent.
        
        Args:
            messages: Input message(s) - can be a string, list of messages, or state dict
            config: Configuration including thread_id for checkpointing
            
        Returns:
            Final state with all messages
        """
        # Reset iteration counter
        self._iteration_count = 0
        
        # Normalize input
        if isinstance(messages, str):
            input_state = {"messages": [HumanMessage(content=messages)]}
        elif isinstance(messages, list):
            input_state = {"messages": messages}
        else:
            input_state = messages
        
        # Invoke the graph
        result = self.graph.invoke(input_state, config=config)
        return result
    
    async def astream_events(
        self,
        messages: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Stream events from the agent execution."""
        self._iteration_count = 0
        
        if isinstance(messages, str):
            input_state = {"messages": [HumanMessage(content=messages)]}
        elif isinstance(messages, list):
            input_state = {"messages": messages}
        else:
            input_state = messages
        
        async for event in self.graph.astream_events(input_state, config=config, **kwargs):
            yield event
    
    async def aget_state(self, config: Dict[str, Any]):
        """Get the current state for a thread."""
        return await self.graph.aget_state(config)
    
    def get_graph_diagram(self) -> str:
        """Get a visual representation of the graph."""
# Convenience functions for quick usage

async def create_react_agent_with_mcp(
    model: str = "gpt-4o",
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
    use_sqlite: bool = False,
    max_iterations: int = 15,
    include_todo_tools: bool = True,
    include_shell_tool: bool = True
) -> tuple:
    """Create a ReACT agent with MCP tools.
    
    Args:
        model: The model to use (default: gpt-4o)
        temperature: Model temperature
        system_prompt: Optional custom system prompt
        use_sqlite: Whether to use SQLite for checkpointing
        max_iterations: Maximum reasoning iterations
        include_todo_tools: Whether to include todo list tools
        include_shell_tool: Whether to include shell execution tool
        
    Returns:
        Tuple of (agent, runner, mcp_manager)
    """
async def quick_chat(message: str, model: str = "gpt-4o") -> str:
    """Quick one-shot chat with the agent.
    
    Args:
        message: The message to send
        model: The model to use
        
    Returns:
        The agent's response as a string
    """
    agent, runner, mcp_manager = await create_react_agent_with_mcp(model=model)
    try:
        result = await runner.run(message)
        # Extract the final AI message
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return "No response generated"
    finally:
        await mcp_manager.disconnect()
        max_iterations=max_iterations,
        use_sqlite_checkpoint=use_sqlite
    )
async def demo():
    """Demonstrate the ReACT agent capabilities."""
    print("=" * 60)
    print("LangGraph ReACT Agent with MCP Tools Demo")
    print("=" * 60)
    
    try:
        # Create the agent with MCP tools
        agent, runner, mcp_manager = await create_react_agent_with_mcp(
            model="gpt-4o",
            temperature=0.1
        )
        
        print("\n✅ Agent created successfully!")
        print(f"📦 Model: {agent.model_name}")
        print(f"📦 Loaded {len(agent.tools)} tools (including MCP tools)")
        print(f"📊 Graph structure: START -> agent <-> tools -> END")
        
        # Example interactions
        print("\n" + "-" * 40)
        print("Example 1: List workspace files")
        print("-" * 40)
        
        result = await runner.run("List the Python files in the workspace root directory")
        
        # Print the response
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n🤖 Assistant: {msg.content[:500]}...")
                break
        
        print("\n" + "-" * 40)
        print("Example 2: Continue conversation (same thread)")
        print("-" * 40)
        
        result = await runner.run("What's in the tools folder?")
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n🤖 Assistant: {msg.content[:500]}...")
                break
        
        print("\n" + "-" * 40)
        print("Example 3: New thread with different topic")
        print("-" * 40)
        
        new_tid = runner.new_thread()
        result = await runner.run(
            "Search for information about LangGraph agents",
            thread_id=new_tid
        )
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n🤖 Assistant: {msg.content[:500]}...")
                break
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        # Cleanup
        await mcp_manager.disconnect()
        print("\n✅ Cleanup completed")ailed: {e}")
            raise
    
    def new_thread(self) -> int:
        """Start a new conversation thread."""
        self._thread_id += 1
        logger.info(f"Started new thread: {self._thread_id}")
        return self._thread_id
    
    async def get_history(self, thread_id: Optional[int] = None) -> List[BaseMessage]:
        """Get conversation history for a thread."""
        config = self._get_config(thread_id)
        state = await self.agent.aget_state(config)
        return state.values.get("messages", [])
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.mcp_manager:
            await self.mcp_manager.disconnect()


# Convenience functions for quick usage

async def create_react_agent_with_mcp(
    model_name: str = "gpt-4o",
    system_prompt: Optional[str] = None,
    use_sqlite: bool = False
) -> tuple:
    """Create a ReACT agent with MCP tools.
    
    Args:
        model_name: The model to use (default: gpt-4o)
        system_prompt: Optional custom system prompt
        use_sqlite: Whether to use SQLite for checkpointing
        
    Returns:
        Tuple of (agent, runner)
    """
    builder = ReACTAgentBuilder(
        model_name=model_name,
        use_sqlite_checkpoint=use_sqlite
    )
    agent = await builder.build_with_mcp(system_prompt)
    runner = ReACTAgentRunner(agent, builder.mcp_manager)
    return agent, runner


async def quick_chat(message: str, model_name: str = "gpt-4o") -> str:
    """Quick one-shot chat with the agent.
    
    Args:
        message: The message to send
        model_name: The model to use
        
    Returns:
        The agent's response as a string
    """
    agent, runner = await create_react_agent_with_mcp(model_name)
    try:
        result = await runner.run(message)
        # Extract the final AI message
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return "No response generated"
    finally:
        await runner.cleanup()


# Example usage and demonstration
async def demo():
    """Demonstrate the ReACT agent capabilities."""
    print("=" * 60)
    print("LangGraph ReACT Agent with MCP Tools Demo")
    print("=" * 60)
    
    # Create the agent
    builder = ReACTAgentBuilder(
        model_name="gpt-4o",
        temperature=0.1,
        use_sqlite_checkpoint=False
    )
    
    try:
        # Build agent with MCP tools and middleware
        agent = await builder.build_with_mcp()
        runner = ReACTAgentRunner(agent, builder.mcp_manager)
        
        print("\n✅ Agent created successfully!")
        print(f"📦 Loaded {len(builder.mcp_manager.tools)} MCP tools")
        
        # Example interactions
        print("\n" + "-" * 40)
        print("Example 1: List workspace files")
        print("-" * 40)
        
        result = await runner.run("List the Python files in the workspace root directory")
        
        # Print the response
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage):
                print(f"\n🤖 Assistant: {msg.content[:500]}...")
                break
        
        print("\n" + "-" * 40)
        print("Example 2: Continue conversation (same thread)")
        print("-" * 40)
        
        result = await runner.run("What's in the tools folder?")
async def interactive_chat():
    """Run an interactive chat session with the agent."""
    print("=" * 60)
    print("LangGraph ReACT Agent - Interactive Chat")
    print("=" * 60)
    print("\nCommands:")
    print("  /new    - Start a new conversation")
    print("  /stream - Toggle streaming mode")
    print("  /quit   - Exit the chat")
    print("=" * 60)
    
    try:
        agent, runner, mcp_manager = await create_react_agent_with_mcp(model="gpt-4o")
        
        print(f"\n✅ Agent ready with {len(agent.tools)} tools")
        print(f"📊 Graph: {agent.graph}")
        
        streaming = False
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "/quit":
                    print("Goodbye!")
                    break
                
                if user_input.lower() == "/new":
                    tid = runner.new_thread()
                    print(f"Started new conversation (thread {tid})")
                    continue
                
                if user_input.lower() == "/stream":
                    streaming = not streaming
                    print(f"Streaming mode: {'ON' if streaming else 'OFF'}")
                    continue
                
                print("\n🤖 Assistant: ", end="")
                
                if streaming:
                    await runner.run(user_input, stream=True)
                    print()  # New line after streaming
                else:
                    result = await runner.run(user_input)
                    for msg in result.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content:
                            print(msg.content)
                            break
                            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
                
    finally:
        await mcp_manager.disconnect()bye!")
                    break
                
                if user_input.lower() == "/new":
                    tid = runner.new_thread()
                    print(f"Started new conversation (thread {tid})")
                    continue
                
                if user_input.lower() == "/stream":
                    streaming = not streaming
                    print(f"Streaming mode: {'ON' if streaming else 'OFF'}")
                    continue
        print("LangGraph ReACT Agent with MCP Tools")
        print("\nUsage:")
        print("  python langgraph_react_agent.py --demo        Run demonstration")
        print("  python langgraph_react_agent.py --interactive Start interactive chat")
        print("\nOr import and use programmatically:")
        print("  from langgraph_react_agent import ReACTAgent, create_react_agent_with_mcp")
        print("\n  # Method 1: Direct instantiation")
        print("  agent = ReACTAgent(model='gpt-4o', temperature=0.1)")
        print("  result = await agent.ainvoke('Your message')")
        print("\n  # Method 2: With MCP tools")
        print("  agent, runner, mcp_manager = await create_react_agent_with_mcp()")
        print("  result = await runner.run('Your message here')")
                    for msg in result.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content:
                            print(msg.content)
                            break
                            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
                
    finally:
        await builder.cleanup()


# Main entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_chat())
    elif len(sys.argv) > 1 and sys.argv[1] == "--demo":
        asyncio.run(demo())
    else:
        print("LangGraph ReACT Agent with MCP Tools")
        print("\nUsage:")
        print("  python langgraph_react_agent.py --demo        Run demonstration")
        print("  python langgraph_react_agent.py --interactive Start interactive chat")
        print("\nOr import and use programmatically:")
        print("  from langgraph_react_agent import create_react_agent_with_mcp")
        print("  agent, runner = await create_react_agent_with_mcp()")
        print("  result = await runner.run('Your message here')")
