from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from llama_stack_client import LlamaStackClient
import os
import httpx
import asyncio
from fastapi.responses import StreamingResponse
import json
import asyncio
import anyio
import threading
import queue

app = FastAPI(title="Canopy Backend API")

LLAMASTACK_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://llamastack-server-genaiops-playground.apps.dev.rhoai.rh-aiservices-bu.com")
CLOUD_LLM = os.getenv("CLOUD_LLM", "llama32-quantized")
PRIVATE_LLM = os.getenv("PRIVATE_LLM", "llama32-quantized")
llama_client = LlamaStackClient(base_url=LLAMASTACK_BASE_URL)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/summarize")
async def summarize(request: PromptRequest):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)