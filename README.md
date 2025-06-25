# canopy-be

FastAPI backend service that uses Llamastack to interact with language models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### POST /generate

Generate text using the language model.

Request body:
```json
{
    "prompt": "Your prompt here",
    "max_tokens": 100,
    "temperature": 0.7
}
```

Response:
```json
{
    "response": "Generated text response",
    "tokens_used": 42
}
```

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`