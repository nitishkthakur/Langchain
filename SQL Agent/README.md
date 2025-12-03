# SQL Agent

A simple Python project that provides a basic web UI chatbot and uses LangChain with Ollama (gemma3:4b) as the LLM.

Features:
- FastAPI backend
- LangChain integration using the Ollama LLM
- Simple static HTML/CSS/JS frontend for chat

Requirements:
- Python 3.10+
- Ollama running locally with gemma3:4b model pulled

Run:
1. python -m venv venv
2. source venv/bin/activate  # or venv\Scripts\activate on Windows
3. pip install -r requirements.txt
4. Start Ollama and ensure gemma3:4b is available: `ollama list`
5. uvicorn app.main:app --reload
6. Open http://127.0.0.1:8000 in your browser
