"""
Tavily Search Tool for LangChain Agents

This module provides a Tavily search tool that can be used with LangChain agents
to perform web searches and retrieve relevant information.

Tavily is a search engine optimized for LLMs and RAG applications, providing
high-quality, relevant search results.

Usage:
    from tools.tavily_search import get_tavily_search_tool
    
    # Get the tool
    search_tool = get_tavily_search_tool()
    
    # Use with an agent
    tools = [search_tool]
    agent = create_react_agent(llm, tools, prompt)

Requirements:
    - tavily-python package installed
    - TAVILY_API_KEY environment variable set

Example:
    ```python
    import os
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_react_agent, AgentExecutor
    from tools.tavily_search import get_tavily_search_tool
    
    # Set up API keys
    os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Get search tool
    search_tool = get_tavily_search_tool()
    
    # Create agent with the tool
    tools = [search_tool]
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Run the agent
    result = agent_executor.invoke({"input": "What is the latest news about AI?"})
    ```
"""

from typing import Optional, Type
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import BaseTool


def get_tavily_search_tool(
    max_results: int = 5,
    search_depth: str = "advanced",
    include_answer: bool = True,
    include_raw_content: bool = False,
    include_images: bool = False,
) -> BaseTool:
    """
    Get a configured Tavily search tool for use with LangChain agents.
    
    This tool allows agents to search the web using Tavily's AI-optimized search engine.
    Tavily provides high-quality, relevant results specifically designed for LLM consumption.
    
    Args:
        max_results (int): Maximum number of search results to return. Default is 5.
        search_depth (str): Depth of search - "basic" or "advanced". Default is "advanced".
            - "basic": Faster, returns quick results
            - "advanced": More thorough, better quality results
        include_answer (bool): Whether to include a generated answer. Default is True.
        include_raw_content (bool): Whether to include raw page content. Default is False.
        include_images (bool): Whether to include images in results. Default is False.
    
    Returns:
        BaseTool: A configured Tavily search tool ready to be used with agents.
    
    Raises:
        ValueError: If TAVILY_API_KEY environment variable is not set.
    
    Example:
        >>> search_tool = get_tavily_search_tool(max_results=3, search_depth="basic")
        >>> # Use this tool with your LangChain agent
    
    Note:
        You must set the TAVILY_API_KEY environment variable before using this tool.
        Get your API key from: https://tavily.com/
    """
    tool = TavilySearchResults(
        max_results=max_results,
        search_depth=search_depth,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
        include_images=include_images,
    )
    
    # Set a more descriptive name and description for the agent
    tool.name = "tavily_search"
    tool.description = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events, "
        "find factual information, or search for specific content on the web. "
        "Input should be a search query string."
    )
    
    return tool


def get_tavily_answer_tool() -> BaseTool:
    """
    Get a Tavily search tool configured to return direct answers.
    
    This variant is optimized to return concise, direct answers to queries,
    making it ideal for question-answering tasks.
    
    Returns:
        BaseTool: A Tavily search tool configured for answer generation.
    
    Example:
        >>> answer_tool = get_tavily_answer_tool()
        >>> # This tool will return direct answers to questions
    """
    return get_tavily_search_tool(
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )


# Tool metadata for agent discovery
__all__ = [
    "get_tavily_search_tool",
    "get_tavily_answer_tool",
]

TOOL_INFO = {
    "name": "Tavily Search",
    "description": "Web search tool optimized for LLM agents",
    "category": "search",
    "requires_api_key": True,
    "api_key_env_var": "TAVILY_API_KEY",
}
