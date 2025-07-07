from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from llama_stack_client import LlamaStackClient
import os
import asyncio
from fastapi.responses import StreamingResponse
import json
import threading
import queue
import yaml

app = FastAPI(title="Canopy Backend API")

config_path = "/canopy/canopy-config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

llama_client = LlamaStackClient(base_url=config["LLAMA_STACK_URL"])

# Feature flags configuration from environment variables
FEATURE_FLAGS = {
    "RAG": "RAG" in config and config["RAG"]["enabled"] == True,
    "summarize": "summarize" in config and config["summarize"]["enabled"] == True,
}

class PromptRequest(BaseModel):
    prompt: str

@app.get("/feature-flags")
async def get_feature_flags() -> Dict[str, Any]:
    """Get all feature flags configuration"""
    return FEATURE_FLAGS

@app.post("/summarize")
async def summarize(request: PromptRequest):
    # Check if summarization feature is enabled
    if not FEATURE_FLAGS.get("summarize", False):
        raise HTTPException(status_code=404, detail="Summarization feature is not enabled")

    sys_prompt = config["summarize"].get("prompt", "Summarize the following text:")
    temperature = config["summarize"].get("temperature", 0.7)
    max_tokens = config["summarize"].get("max_tokens", 4096)

    q = queue.Queue()

    def worker():
        print(f"sending requestion to model {config['summarize']['model']}")
        try:
            response = llama_client.inference.chat_completion(
                model_id=config["summarize"]["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": request.prompt},
                ],
                sampling_params={"max_tokens": max_tokens, "temperature": temperature},
                stream=True,
            )
            for r in response:
                if hasattr(r.event, 'delta') and hasattr(r.event.delta, 'text'):
                    chunk = f"data: {json.dumps({'delta': r.event.delta.text})}\n\n"
                    q.put(chunk)
        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        finally:
            q.put(None)

    threading.Thread(target=worker).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

@app.post("/rag")
async def rag(request: PromptRequest):
    # Check if RAG feature is enabled
    if not FEATURE_FLAGS.get("RAG", False):
        raise HTTPException(status_code=404, detail="RAG feature is not enabled")
    
    # Dummy RAG implementation
    sys_prompt = config["RAG"]["prompt"]

    q = queue.Queue()

    def worker():
        try:
            response = llama_client.inference.chat_completion(
                model_id=config["RAG"]["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"[RAG Context] Answer this question: {request.prompt}"},
                ],
                sampling_params={"max_tokens": 4096, "temperature": 0.7},
                stream=True,
            )
            for r in response:
                if hasattr(r.event, 'delta') and hasattr(r.event.delta, 'text'):
                    chunk = f"data: {json.dumps({'delta': r.event.delta.text})}\n\n"
                    q.put(chunk)
        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")

    threading.Thread(target=worker).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)