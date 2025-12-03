"""
Example usage of the custom ReACT Agent with LangGraph nodes and edges.

This demonstrates how to:
1. Create a ReACT agent with custom model and prompt
2. Use the agent with MCP tools
3. Customize the agent configuration
"""

import asyncio
from langgraph_react_agent import ReACTAgent, create_react_agent_with_mcp, MCPToolManager, MCP_SERVER_CONFIGS


async def example_1_basic_agent():
    """Example 1: Create a basic ReACT agent without MCP tools."""
    print("=" * 60)
    print("Example 1: Basic ReACT Agent")
    print("=" * 60)
    
    # Create agent with custom configuration
    agent = ReACTAgent(
        model="gpt-4o-mini",  # Specify model
        temperature=0.7,       # Set temperature
        system_prompt="You are a helpful math tutor. Explain concepts clearly.",
        tools=[],              # No tools for this simple example
        max_iterations=10
    )
    
    # Use the agent
    result = await agent.ainvoke("What is 15 * 23?")
    
    # Get the response
    messages = result["messages"]
    for msg in messages:
        if hasattr(msg, 'content') and msg.content:
            print(f"\n{msg.__class__.__name__}: {msg.content}")
    
    print("\n✅ Example 1 complete")


async def example_2_agent_with_mcp():
    """Example 2: ReACT Agent with MCP tools."""
    print("\n" + "=" * 60)
    print("Example 2: ReACT Agent with MCP Tools")
    print("=" * 60)
    
    # Create agent with MCP tools using helper function
    agent, runner, mcp_manager = await create_react_agent_with_mcp(
        model="gpt-4o-mini",
        temperature=0.1,
        system_prompt="You are a helpful coding assistant with filesystem access.",
        include_todo_tools=True,
        include_shell_tool=True
    )
    
    try:
        print(f"\n📦 Created agent with {len(agent.tools)} tools")
        print(f"📊 Max iterations: {agent.max_iterations}")
        print(f"🤖 Model: {agent.model_name}")
        
        # Use the agent
        result = await runner.run("List all Python files in the current directory")
        
        # Display response
        messages = result.get("messages", [])
        for msg in messages:
            if hasattr(msg, 'content') and msg.content and msg.__class__.__name__ == "AIMessage":
                print(f"\n🤖 Response: {msg.content[:300]}...")
                break
        
        print("\n✅ Example 2 complete")
        
    finally:
        await mcp_manager.disconnect()


async def example_3_custom_configuration():
    """Example 3: Fully customized ReACT Agent."""
    print("\n" + "=" * 60)
    print("Example 3: Customized ReACT Agent")
    print("=" * 60)
    
    # Connect to MCP servers manually
    mcp_manager = MCPToolManager(MCP_SERVER_CONFIGS)
    await mcp_manager.connect()
    
    try:
        # Create agent with custom configuration
        agent = ReACTAgent(
            model="gpt-4o",
            temperature=0.0,  # Deterministic
            system_prompt="""You are a specialized code analysis agent.
            
Your capabilities:
- Analyze code structure and quality
- Review code for best practices
- Suggest improvements

Always be thorough and provide actionable feedback.""",
            tools=mcp_manager.tools,
            max_iterations=20,  # Allow more iterations for complex tasks
            use_sqlite_checkpoint=False
        )
        
        print(f"\n📦 Agent configuration:")
        print(f"   Model: {agent.model_name}")
        print(f"   Temperature: {agent.temperature}")
        print(f"   Tools: {len(agent.tools)}")
        print(f"   Max iterations: {agent.max_iterations}")
        
        # Test with a simple query
        result = await agent.ainvoke("What files are in the tools directory?")
        
        messages = result["messages"]
        for msg in messages:
            if hasattr(msg, 'content') and msg.content and msg.__class__.__name__ == "AIMessage":
                print(f"\n🤖 Response: {msg.content[:400]}...")
                break
        
        print("\n✅ Example 3 complete")
        
    finally:
        await mcp_manager.disconnect()


async def example_4_checkpointing():
    """Example 4: Agent with conversation memory using checkpointing."""
    print("\n" + "=" * 60)
    print("Example 4: Agent with Checkpointing")
    print("=" * 60)
    
    # Create agent with SQLite checkpointing
    agent, runner, mcp_manager = await create_react_agent_with_mcp(
        model="gpt-4o-mini",
        use_sqlite=True
    )
    
    try:
        print("\n📝 First message...")
        result1 = await runner.run("My favorite color is blue", thread_id=1)
        
        print("\n📝 Second message (same thread)...")
        result2 = await runner.run("What's my favorite color?", thread_id=1)
        
        # Check if it remembered
        messages = result2.get("messages", [])
        for msg in messages:
            if hasattr(msg, 'content') and msg.content and "blue" in msg.content.lower():
                print("✅ Agent remembered the favorite color!")
                print(f"   Response: {msg.content[:200]}...")
                break
        
        print("\n📝 New thread (should not remember)...")
        new_tid = runner.new_thread()
        result3 = await runner.run("What's my favorite color?", thread_id=new_tid)
        
        print("\n✅ Example 4 complete")
        
    finally:
        await mcp_manager.disconnect()


async def example_5_direct_usage():
    """Example 5: Direct usage of ReACT agent without runner."""
    print("\n" + "=" * 60)
    print("Example 5: Direct Agent Usage")
    print("=" * 60)
    
    # Create a simple agent
    agent = ReACTAgent(
        model="gpt-4o-mini",
        temperature=0.5,
        system_prompt="You are a creative writing assistant.",
        tools=[],  # No tools
        max_iterations=5
    )
    
    # Direct invocation
    result = await agent.ainvoke(
        "Write a haiku about programming",
        config={"configurable": {"thread_id": "creative-session"}}
    )
    
    print("\n📝 Haiku:")
    for msg in result["messages"]:
        if hasattr(msg, 'content') and msg.content and msg.__class__.__name__ == "AIMessage":
            print(msg.content)
            break
    
    print("\n✅ Example 5 complete")


async def main():
    """Run all examples."""
    print("🚀 ReACT Agent Examples\n")
    
    # Run examples
    await example_1_basic_agent()
    
    # Only run MCP examples if you have API keys configured
    import os
    if os.getenv("OPENAI_API_KEY"):
        await example_2_agent_with_mcp()
        await example_3_custom_configuration()
        await example_4_checkpointing()
        await example_5_direct_usage()
    else:
        print("\n⚠️  Set OPENAI_API_KEY to run MCP examples")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
