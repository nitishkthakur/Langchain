"""
LangGraph ReACT Agent with Native Graph Architecture

Custom ReACT agent using LangGraph's StateGraph with explicit nodes and edges.
No readymade create_agent - full manual control over the graph structure.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Literal, Union

from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
    SystemMessage
)
from langchain_core.tools import BaseTool, tool

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated

# MCP imports
from langchain_mcp_adapters import MultiServerMCPClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path("/home/nitish/Documents/github/Langchain")
TOOLS_DIR = WORKSPACE_ROOT / "tools"

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
    """State for the ReACT agent graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class ReACTAgent:
    """Custom ReACT Agent using LangGraph's native graph construction."""
    
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
        self.model_name = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.tools = tools or []
        self._iteration_count = 0
        
        self.llm = init_chat_model(model, temperature=temperature, timeout=60.0, max_tokens=4096)
        
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
        
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        if use_sqlite_checkpoint:
            self.checkpointer = SqliteSaver.from_conn_string(sqlite_path)
        else:
            self.checkpointer = InMemorySaver()
        
        self.graph = self._build_graph()
        logger.info(f"ReACT Agent initialized: {model}, {len(self.tools)} tools")
    
    def _default_system_prompt(self) -> str:
        return """You are a helpful AI assistant with access to tools.
        
Be thorough and explain your reasoning. When complete, do NOT call more tools."""
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._execute_tools)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue, {"continue": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        messages = list(state["messages"])
        
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        response = self.llm.invoke(messages)
        self._iteration_count += 1
        
        return {"messages": [response]}
    
    def _execute_tools(self, state: AgentState) -> Dict[str, Any]:
        tool_node = ToolNode(self.tools)
        return tool_node.invoke(state)
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if self._iteration_count >= self.max_iterations:
            return "end"
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        return "end"
    
    async def ainvoke(self, messages, config=None):
        self._iteration_count = 0
        
        if isinstance(messages, str):
            input_state = {"messages": [HumanMessage(content=messages)]}
        elif isinstance(messages, list):
            input_state = {"messages": messages}
        else:
            input_state = messages
        
        return await self.graph.ainvoke(input_state, config=config)
    
    def invoke(self, messages, config=None):
        self._iteration_count = 0
        
        if isinstance(messages, str):
            input_state = {"messages": [HumanMessage(content=messages)]}
        elif isinstance(messages, list):
            input_state = {"messages": messages}
        else:
            input_state = messages
        
        return self.graph.invoke(input_state, config=config)


class MCPToolManager:
    def __init__(self, server_configs):
        self.server_configs = server_configs
        self._client = None
        self._tools = []
        self._connected = False
    
    async def connect(self):
        if self._connected:
            return
        
        self._client = MultiServerMCPClient(self.server_configs)
        await asyncio.wait_for(self._client.connect(), timeout=30.0)
        self._tools = await self._client.get_tools()
        self._connected = True
        logger.info(f"MCP connected: {len(self._tools)} tools")
    
    async def disconnect(self):
        if self._client and self._connected:
            await self._client.disconnect()
            self._connected = False
    
    @property
    def tools(self):
        return self._tools


async def create_react_agent_with_mcp(
    model="gpt-4o",
    temperature=0.1,
    system_prompt=None,
    use_sqlite=False,
    max_iterations=15
):
    mcp_manager = MCPToolManager(MCP_SERVER_CONFIGS)
    await mcp_manager.connect()
    
    agent = ReACTAgent(
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        tools=mcp_manager.tools,
        max_iterations=max_iterations,
        use_sqlite_checkpoint=use_sqlite
    )
    
    return agent, mcp_manager


async def demo():
    print("=" * 60)
    print("ReACT Agent Demo")
    print("=" * 60)
    
    agent, mcp_manager = await create_react_agent_with_mcp()
    
    try:
        print(f"\n✅ Agent ready: {agent.model_name}, {len(agent.tools)} tools")
        
        result = await agent.ainvoke("List Python files in workspace")
        
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n🤖 {msg.content[:300]}...")
                break
    
    finally:
        await mcp_manager.disconnect()


if __name__ == "__main__":
    import sys
    
    if "--demo" in sys.argv:
        asyncio.run(demo())
    else:
        print("Usage: python langgraph_react_agent_new.py --demo")
