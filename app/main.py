from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from llama_stack_client import LlamaStackClient
import os
import httpx
import asyncio
from fastapi.responses import StreamingResponse
import json
import anyio
import threading
import queue

app = FastAPI(title="Canopy Backend API")

LLAMASTACK_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://llamastack-server-genaiops-playground.apps.dev.rhoai.rh-aiservices-bu.com")
CLOUD_LLM = os.getenv("CLOUD_LLM", "llama32-quantized")
PRIVATE_LLM = os.getenv("PRIVATE_LLM", "llama32-quantized")
llama_client = LlamaStackClient(base_url=LLAMASTACK_BASE_URL)

# Feature flags configuration from environment variables
FEATURE_FLAGS = {
    "rag-feature": os.getenv("FEATURE_RAG_ENABLED", "false").lower() == "true",
    "summarization": os.getenv("FEATURE_SUMMARIZATION_ENABLED", "true").lower() == "true",
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
    if not FEATURE_FLAGS.get("summarization", False):
        raise HTTPException(status_code=404, detail="Summarization feature is not enabled")
    
    sys_prompt = "Give me a good summary of the following text."

    q = queue.Queue()

    def worker():
        try:
            response = llama_client.inference.chat_completion(
                model_id=CLOUD_LLM,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": request.prompt},
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

@app.post("/rag")
async def rag(request: PromptRequest):
    # Check if RAG feature is enabled
    if not FEATURE_FLAGS.get("rag-feature", False):
        raise HTTPException(status_code=404, detail="RAG feature is not enabled")
    
    # Dummy RAG implementation
    sys_prompt = "You are a helpful assistant that provides answers based on retrieved context."
    
    q = queue.Queue()

    def worker():
        try:
            response = llama_client.inference.chat_completion(
                model_id=CLOUD_LLM,
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