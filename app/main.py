from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from llama_stack_client import LlamaStackClient
import asyncio
from fastapi.responses import StreamingResponse
import json
import threading
import queue
import yaml

app = FastAPI(title="Canopy Backend API")

# config_path = "/canopy/canopy-config.yaml"
config_path = "/Users/ckavili/RedHat/Tech/AI500/GenAIOps/canopy-be/app/canopy-config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

llama_client = LlamaStackClient(base_url=config["LLAMA_STACK_URL"])

# Feature flags configuration from environment variables
FEATURE_FLAGS = {
    "information-search": "information-search" in config and config["information-search"]["enabled"] == True,
    "summarize": "summarize" in config and config["summarize"]["enabled"] == True,
}

# Shield configuration
SHIELDS_ENABLED = config.get("shields", {}).get("enabled", False)
INPUT_SHIELDS = config.get("shields", {}).get("input_shields", [])
OUTPUT_SHIELDS = config.get("shields", {}).get("output_shields", [])

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
            if SHIELDS_ENABLED:
                # Use agents API when shields are enabled
                agent_config = {
                    "model": config["summarize"]["model"],
                    "instructions": sys_prompt,
                    "sampling_params": {"max_tokens": max_tokens, "temperature": temperature},
                }
                if INPUT_SHIELDS:
                    agent_config["input_shields"] = INPUT_SHIELDS
                if OUTPUT_SHIELDS:
                    agent_config["output_shields"] = OUTPUT_SHIELDS

                # Create agent
                agent_response = llama_client.agents.create(agent_config=agent_config)
                agent_id = agent_response.agent_id

                # Create session
                session_response = llama_client.agents.session.create(
                    agent_id=agent_id,
                    session_name="summarize_session"
                )
                session_id = session_response.session_id

                # Send turn with streaming
                response = llama_client.agents.turn.create(
                    agent_id=agent_id,
                    session_id=session_id,
                    messages=[{"role": "user", "content": request.prompt}],
                    stream=True,
                )

                for r in response:
                    # Extract text from agents API streaming response
                    text_content = None
                    error_content = None

                    # Agents API uses event.payload.delta.text for step_progress events
                    if hasattr(r, 'event') and hasattr(r.event, 'payload'):
                        payload = r.event.payload

                        # Check for step_progress event with delta (normal streaming)
                        if hasattr(payload, 'event_type') and payload.event_type == 'step_progress':
                            if hasattr(payload, 'delta') and hasattr(payload.delta, 'text'):
                                text_content = payload.delta.text

                        # Check for shield violations (step_complete with violation)
                        elif hasattr(payload, 'event_type') and payload.event_type == 'step_complete':
                            if hasattr(payload, 'step_details'):
                                step_details = payload.step_details
                                if hasattr(step_details, 'step_type') and step_details.step_type == 'shield_call':
                                    if hasattr(step_details, 'violation') and step_details.violation is not None:
                                        # User-friendly error message
                                        error_content = "I'm sorry, but I can't help with that request. Please try asking something else."

                    if text_content:
                        chunk = f"data: {json.dumps({'delta': text_content})}\n\n"
                        q.put(chunk)
                    elif error_content:
                        chunk = f"data: {json.dumps({'error': error_content})}\n\n"
                        q.put(chunk)
                        break  # Stop processing after a violation
            else:
                # Use inference API when shields are disabled
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

    # vector_db_id = "docling_vector_db_genaiops" # Hardcoded as we always will use this collection for this usecase
    rag_response = llama_client.tool_runtime.rag_tool.query(
        content=request.prompt,                               # User's question
        vector_db_ids=[vector_db_id],               # Document intelligence database
        query_config={                              # Format retrieved results
            "chunk_template": "Result {index}\\nContent: {chunk.content}\\nMetadata: {metadata}\\n",
        },
    )
    prompt_context = rag_response.content

    enhaned_prompt = f"""Please answer the given query using the document intelligence context below.

    CONTEXT (Processed with Docling Document Intelligence):
    {prompt_context}

    QUERY:
    {request.prompt}

    Note: The context includes intelligently processed content with preserved tables, formulas, figures, and document structure."""

    def worker():
        try:
            if SHIELDS_ENABLED:
                # Use agents API when shields are enabled
                agent_config = {
                    "model": config["information-search"]["model"],
                    "instructions": sys_prompt,
                    "sampling_params": {"max_tokens": max_tokens, "temperature": temperature},
                }
                if INPUT_SHIELDS:
                    agent_config["input_shields"] = INPUT_SHIELDS
                if OUTPUT_SHIELDS:
                    agent_config["output_shields"] = OUTPUT_SHIELDS

                # Create agent
                agent_response = llama_client.agents.create(agent_config=agent_config)
                agent_id = agent_response.agent_id

                # Create session
                session_response = llama_client.agents.session.create(
                    agent_id=agent_id,
                    session_name="information_search_session"
                )
                session_id = session_response.session_id

                # Send turn with streaming
                response = llama_client.agents.turn.create(
                    agent_id=agent_id,
                    session_id=session_id,
                    messages=[{"role": "user", "content": enhaned_prompt}],
                    stream=True,
                )

                for r in response:
                    # Extract text from agents API streaming response
                    text_content = None
                    error_content = None

                    # Agents API uses event.payload.delta.text for step_progress events
                    if hasattr(r, 'event') and hasattr(r.event, 'payload'):
                        payload = r.event.payload

                        # Check for step_progress event with delta (normal streaming)
                        if hasattr(payload, 'event_type') and payload.event_type == 'step_progress':
                            if hasattr(payload, 'delta') and hasattr(payload.delta, 'text'):
                                text_content = payload.delta.text

                        # Check for shield violations (step_complete with violation)
                        elif hasattr(payload, 'event_type') and payload.event_type == 'step_complete':
                            if hasattr(payload, 'step_details'):
                                step_details = payload.step_details
                                if hasattr(step_details, 'step_type') and step_details.step_type == 'shield_call':
                                    if hasattr(step_details, 'violation') and step_details.violation is not None:
                                        # User-friendly error message
                                        error_content = "I'm sorry, but I can't help with that request. Please try asking something else."

                    if text_content:
                        chunk = f"data: {json.dumps({'delta': text_content})}\n\n"
                        q.put(chunk)
                    elif error_content:
                        chunk = f"data: {json.dumps({'error': error_content})}\n\n"
                        q.put(chunk)
                        break  # Stop processing after a violation
            else:
                # Use inference API when shields are disabled
                response = llama_client.inference.chat_completion(
                    model_id=config["information-search"]["model"],
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": enhaned_prompt},
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)