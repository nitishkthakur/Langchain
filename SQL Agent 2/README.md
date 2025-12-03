SQL Agent 2

A minimal chat UI backed by LangChain's Ollama LLM (gemma3:4b).

Run:

1. pip install -r requirements.txt
2. Ensure ollama daemon is running locally (https://ollama.com/) and has gemma3:4b. You can check available models by running `ollama list` or visiting `/models` endpoint of this app.

3. uvicorn app:app --reload
4. Open http://localhost:8000
