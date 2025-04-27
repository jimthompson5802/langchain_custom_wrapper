# LangChain OpenAI API Wrapper

This project provides a FastAPI server that wraps the LangChain OpenAI chat completion API.

## Setup

1. Make sure you have all the required dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY="your_openai_api_key"
   ```

## Running the Server

To start the server, run:

```bash
cd /Users/jim/Desktop/genai/langchain_wrapper
python src/langchain_server.py
```

The server will be available at `http://localhost:8000`.

## API Endpoints

### POST /v1/chat/completions

This endpoint accepts chat completion requests and returns the generated response using LangChain's OpenAI integration.

#### Request Body

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of Hawaii?"
    }
  ],
  "model": "gpt-3.5-turbo",
  "temperature": 0.7,
  "max_tokens": null
}
```

#### Response

```json
{
  "content": "The capital of Hawaii is Honolulu, located on the island of Oahu.",
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 15,
    "total_tokens": 43
  },
  "additional_kwargs": {}
}
```

### GET /health

Health check endpoint that returns the status of the API.

## Interactive API Documentation

FastAPI automatically generates interactive API documentation. After starting the server, you can access:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc