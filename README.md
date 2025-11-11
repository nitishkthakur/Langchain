# Langchain
Agentic Work using Langchain

This repository contains examples and implementations for building AI agents using LangChain.

## 📁 Project Structure

```
.
├── agents/              # Custom agent implementations
├── memory/              # Memory implementations for agents
├── retrievers/          # Document retrievers
├── tools/               # Tools for agents (e.g., Tavily search)
│   └── tavily_search.py # Web search tool
├── 00.react_agent.ipynb        # ReAct agent examples
├── 01.deep_agents_basic.ipynb  # Deep agents with memory
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/nitishkthakur/Langchain.git
cd Langchain

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [Tavily](https://tavily.com/)

### 4. Run the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Open and run:
- `00.react_agent.ipynb` - Learn about ReAct agents
- `01.deep_agents_basic.ipynb` - Explore deep agents with memory

## 📚 What's Included

### Tools

#### Tavily Search Tool (`tools/tavily_search.py`)
A web search tool optimized for LLM agents. Features:
- Advanced web search capabilities
- Configurable search depth and results
- Direct answer generation
- Well-documented API for easy integration

**Example usage:**
```python
from tools.tavily_search import get_tavily_search_tool

search_tool = get_tavily_search_tool(max_results=5, search_depth="advanced")
```

### Notebooks

#### 00. ReAct Agent (`00.react_agent.ipynb`)
Learn how to build and use ReAct (Reasoning and Acting) agents:
- Setting up a basic agent
- Integrating tools
- Understanding the ReAct loop
- Practical examples

#### 01. Deep Agents Basic (`01.deep_agents_basic.ipynb`)
Explore advanced agent patterns:
- Agents with memory
- Multi-turn conversations
- Complex problem decomposition
- Self-reflection and planning

## 🧰 Available Tools

| Tool | Description | Location |
|------|-------------|----------|
| Tavily Search | Web search optimized for LLMs | `tools/tavily_search.py` |

## 📦 Dependencies

Key dependencies (see `requirements.txt` for full list):
- `langchain` - Core LangChain framework
- `langchain-openai` - OpenAI integration
- `tavily-python` - Tavily search API
- `jupyter` - Jupyter notebooks
- `python-dotenv` - Environment variable management

## 🎯 Use Cases

This repository demonstrates:
- Building conversational AI agents
- Implementing web search capabilities
- Creating agents with memory
- Handling multi-step reasoning tasks
- Integrating multiple tools

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new tools
- Create additional examples
- Improve documentation
- Report issues

## 📄 License

This project is open source. Please check the license file for details.

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [Tavily Search](https://docs.tavily.com/)
- [LangSmith](https://docs.smith.langchain.com/)

## 💡 Tips

- Start with `00.react_agent.ipynb` if you're new to agents
- Use `.env` file to manage API keys securely
- Check the tools documentation for integration examples
- Experiment with different LLM models and parameters

## ⚠️ Important Notes

- Never commit your `.env` file with actual API keys
- The `.gitignore` is configured to exclude sensitive files
- API usage may incur costs - monitor your usage
- Always test with small examples first
