from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from llama_stack import LlamaClient

app = FastAPI(title="Canopy Backend API")
llama_client = LlamaClient()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

class PromptResponse(BaseModel):
    response: str
    tokens_used: Optional[int]

@app.post("/generate", response_model=PromptResponse)
async def generate_response(request: PromptRequest):
    try:
        response = await llama_client.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return PromptResponse(
            response=response.text,
            tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)