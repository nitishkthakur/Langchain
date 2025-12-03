from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import json
import requests
from dotenv import load_dotenv

# LangChain + Ollama
from langchain.llms import Ollama

load_dotenv()
app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

# Config - model
MODEL_NAME = os.getenv('OLLAMA_MODEL', 'gemma3:4b')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

class ChatRequest(BaseModel):
    message: str

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/chat')
async def chat(req: ChatRequest):
    try:
        llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
        # generate
        resp = llm(req.message)
        return JSONResponse({'reply': str(resp)})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get('/models')
async def models():
    """Call the local Ollama HTTP API to list models."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/models")
        r.raise_for_status()
        return JSONResponse(r.json())
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)
