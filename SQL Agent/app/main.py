from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from langchain.llms import Ollama

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="./static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama LLM wrapper; endpoint defaults (ollama must be running locally)
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma3:4b")
llm = Ollama(model=MODEL_NAME)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/models")
async def list_models():
    # Ollama library might not expose list; use shell via environment or client if available
    try:
        # Ollama Python client has 'list' method in some versions
        models = llm.client.list_models() if hasattr(llm, 'client') else [MODEL_NAME]
    except Exception:
        models = [MODEL_NAME]
    return JSONResponse({"models": models})

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    try:
        resp = llm(message)
        return JSONResponse({"response": resp})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
