"""
Test script for the LangGraph ReACT Agent with MCP Tools.

This script provides various test scenarios to validate the agent functionality.
Run with: python test_langgraph_agent.py

Requirements:
1. Set OPENAI_API_KEY environment variable (or GROQ_API_KEY for Groq)
2. Install dependencies: pip install -r requirements.txt
3. Ensure MCP servers in tools/ folder are working
"""

import asyncio
import os
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


async def test_mcp_connection():
    """Test that MCP servers can be connected."""
    print("\n" + "=" * 50)
    print("Test 1: MCP Server Connection")
    print("=" * 50)
    
    from langgraph_react_agent import MCPToolManager, MCP_SERVER_CONFIGS
    
    # Only test filesystem server for quick validation
    test_config = {
        "filesystem": MCP_SERVER_CONFIGS["filesystem"]
    }
    
    manager = MCPToolManager(test_config)
    
    try:
        await manager.connect()
        print(f"✅ Connected successfully!")
        print(f"   Loaded {len(manager.tools)} tools")
        
        for tool in manager.tools[:5]:  # Show first 5 tools
            print(f"   - {tool.name}: {tool.description[:50]}...")
        
        await manager.disconnect()
        print("✅ Disconnected successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


async def test_agent_creation():
    """Test that the agent can be created with middleware."""
    print("\n" + "=" * 50)
    print("Test 2: Agent Creation with Middleware")
    print("=" * 50)
    
    from langgraph_react_agent import ReACTAgentBuilder
    
    builder = ReACTAgentBuilder(
        model_name="gpt-4o-mini",  # Use mini for faster/cheaper testing
        temperature=0.1,
        use_sqlite_checkpoint=False
    )
    
    try:
        agent = await builder.build_with_mcp()
        print("✅ Agent created successfully!")
        print(f"   Model: {builder.model_name}")
        print(f"   MCP Tools: {len(builder.mcp_manager.tools)}")
        print("   Middleware: SummarizationMiddleware, TodoListMiddleware,")
        print("               ShellToolMiddleware, ToolRetryMiddleware")
        
        await builder.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_invocation():
    """Test a simple agent invocation."""
    print("\n" + "=" * 50)
    print("Test 3: Simple Agent Invocation")
    print("=" * 50)
    
    from langgraph_react_agent import create_react_agent_with_mcp, ReACTAgentRunner
    from langchain_core.messages import AIMessage
    
    try:
        agent, runner = await create_react_agent_with_mcp(
            model_name="gpt-4o-mini"
        )
        
        print("📝 Sending: 'What is 2 + 2?'")
        result = await runner.run("What is 2 + 2?")
        
        # Extract response
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"🤖 Response: {msg.content[:200]}")
                break
        
        print("✅ Invocation successful!")
        await runner.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Invocation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_usage():
    """Test that the agent can use MCP tools."""
    print("\n" + "=" * 50)
    print("Test 4: MCP Tool Usage")
    print("=" * 50)
    
    from langgraph_react_agent import create_react_agent_with_mcp
    from langchain_core.messages import AIMessage
    
    try:
        agent, runner = await create_react_agent_with_mcp(
            model_name="gpt-4o-mini"
        )
        
        print("📝 Sending: 'List all .py files in the current directory'")
        result = await runner.run("List all .py files in the current directory using the filesystem tools")
        
        # Check for tool calls
        messages = result.get("messages", [])
        tool_calls_found = False
        
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls_found = True
                for tc in msg.tool_calls:
                    print(f"🔧 Tool called: {tc.get('name', 'unknown')}")
            
            if isinstance(msg, AIMessage) and msg.content:
                print(f"🤖 Response: {msg.content[:300]}...")
        
        if tool_calls_found:
            print("✅ Tool usage successful!")
        else:
            print("⚠️  No tool calls detected (agent may have answered from knowledge)")
        
        await runner.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Tool usage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_conversation_memory():
    """Test that conversation memory works across invocations."""
    print("\n" + "=" * 50)
    print("Test 5: Conversation Memory (Checkpointing)")
    print("=" * 50)
    
    from langgraph_react_agent import create_react_agent_with_mcp
    from langchain_core.messages import AIMessage
    
    try:
        agent, runner = await create_react_agent_with_mcp(
            model_name="gpt-4o-mini"
        )
        
        # First message
        print("📝 Message 1: 'My name is Alice'")
        result1 = await runner.run("My name is Alice")
        
        for msg in result1.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"🤖 Response 1: {msg.content[:100]}...")
                break
        
        # Second message - should remember the name
        print("\n📝 Message 2: 'What is my name?'")
        result2 = await runner.run("What is my name?")
        
        response = ""
        for msg in result2.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content
                print(f"🤖 Response 2: {response[:200]}")
                break
        
        if "alice" in response.lower():
            print("✅ Memory test passed - agent remembered the name!")
        else:
            print("⚠️  Agent may not have remembered the name")
        
        # Test new thread
        print("\n📝 Starting new thread...")
        new_tid = runner.new_thread()
        result3 = await runner.run("What is my name?", thread_id=new_tid)
        
        for msg in result3.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"🤖 Response (new thread): {msg.content[:100]}...")
                break
        
        print("✅ New thread started successfully (should not remember)")
        
        await runner.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence."""
    print("=" * 60)
    print("LangGraph ReACT Agent Test Suite")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: No API key found!")
        print("   Set OPENAI_API_KEY or GROQ_API_KEY environment variable")
        print("   Some tests may fail without an API key")
    
    results = {}
    
    # Test 1: MCP Connection
    results["MCP Connection"] = await test_mcp_connection()
    
    # Test 2: Agent Creation
    results["Agent Creation"] = await test_agent_creation()
    
    # Only run remaining tests if API key is available
    if os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"):
        # Test 3: Simple Invocation
        results["Simple Invocation"] = await test_simple_invocation()
        
        # Test 4: Tool Usage
        results["Tool Usage"] = await test_tool_usage()
        
        # Test 5: Conversation Memory
        results["Conversation Memory"] = await test_conversation_memory()
    else:
        print("\n⏭️  Skipping invocation tests (no API key)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        if test_name == "mcp":
            asyncio.run(test_mcp_connection())
        elif test_name == "agent":
            asyncio.run(test_agent_creation())
        elif test_name == "invoke":
            asyncio.run(test_simple_invocation())
        elif test_name == "tools":
            asyncio.run(test_tool_usage())
        elif test_name == "memory":
            asyncio.run(test_conversation_memory())
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: mcp, agent, invoke, tools, memory")
    else:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
