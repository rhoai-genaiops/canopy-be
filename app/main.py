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
    "information-search": "information-search" in config and config["information-search"]["enabled"] == True,
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
            response = llama_client.chat.completions.create(
                model=config["summarize"]["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": request.prompt},
                ],
                max_tokens=max_tokens, 
                temperature=temperature,
                stream=True,
            )
            for r in response:
                if hasattr(r, 'choices') and r.choices:
                    delta = r.choices[0].delta
                    chunk = f"data: {json.dumps({'delta': delta.content})}\n\n"
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

@app.post("/information-search")
async def information_search(request: PromptRequest):
    # Check if information search feature is enabled
    if not FEATURE_FLAGS.get("information-search", False):
        raise HTTPException(status_code=404, detail="Information search feature is not enabled")

    # Dummy information search implementation
    sys_prompt = config["information-search"]["prompt"]
    temperature = config["information-search"].get("temperature", 0.7)
    max_tokens = config["information-search"].get("max_tokens", 4096)
    vector_db_id = config["information-search"].get("vector_db_id", "latest")

    q = queue.Queue()

    print(f"Searching in collection {vector_db_id}")
    print(f"Existing collections: {llama_client.vector_stores.list()}")
    print(f"query: {request.prompt}")

    search_results = llama_client.vector_stores.search(
        vector_store_id=vector_db_id,
        query=request.prompt,
        max_num_results=5,
        search_mode="vector"
    )
    retrieved_chunks = []
    for i, result in enumerate(search_results.data):
        chunk_content = result.content[0].text if hasattr(result, 'content') else str(result)
        retrieved_chunks.append(chunk_content)

    prompt_context = "\n\n".join(retrieved_chunks)

    enhaned_prompt = f"""Please answer the given query using the document intelligence context below.

    CONTEXT (Processed with Docling Document Intelligence):
    {prompt_context}

    QUERY:
    {request.prompt}

    Note: The context includes intelligently processed content with preserved tables, formulas, figures, and document structure."""

    def worker():
        print(f"sending requestion to model {config['summarize']['model']}")
        try:
            response = llama_client.chat.completions.create(
                model=config["summarize"]["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": enhaned_prompt},
                ],
                max_tokens=max_tokens, 
                temperature=temperature,
                stream=True,
            )
            for r in response:
                if hasattr(r, 'choices') and r.choices:
                    delta = r.choices[0].delta
                    chunk = f"data: {json.dumps({'delta': delta.content})}\n\n"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)