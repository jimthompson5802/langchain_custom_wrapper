import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

import uuid
import json
import redis
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()

# Inherit existing logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisConfig(BaseModel):
    host: str = Field(os.environ.get("REDIS_HOST", "localhost"), description="Redis host")
    port: int = Field(int(os.environ.get("REDIS_PORT", 6379)), description="Redis port")
    db: int = Field(int(os.environ.get("REDIS_DB", 0)), description="Redis database")
    password: Optional[str] = Field(os.environ.get("REDIS_PASSWORD"), description="Redis password")
    conversation_ttl: int = Field(
        int(os.environ.get("CONVERSATION_TTL", 3600)),
        description="Conversation time to live in seconds",
    )


# Global Redis configuration
redis_config = RedisConfig()


def get_redis_client():
    """Get Redis client instance"""
    try:
        client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
            decode_responses=True,
        )
        # Test connection
        client.ping()
        return client
    except redis.ConnectionError as e:
        raise HTTPException(status_code=500, detail=f"Redis connection error: {str(e)}")


def get_binary_redis_client():
    """Get Redis client instance without decode_responses for binary data (like model storage)"""
    try:
        client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
        )
        # Test connection
        client.ping()
        return client
    except redis.ConnectionError as e:
        raise HTTPException(status_code=500, detail=f"Redis connection error: {str(e)}")


app = FastAPI(
    title="LangChain OpenAI API Wrapper",
    description="A FastAPI server that wraps the LangChain OpenAI chat completion API"
    " with Redis state management",
    version="1.0.0",
)


class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (system, user, assistant)")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    model: str = Field("gpt-3.5-turbo", description="The OpenAI model to use")
    temperature: float = Field(0.7, description="Controls randomness of the output")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for continuing a conversation"
    )


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    content: str = Field(..., description="The generated text from the model")
    conversation_id: str = Field(..., description="The conversation ID for future reference")
    usage: Optional[TokenUsage] = Field(None, description="Token usage information")
    additional_kwargs: Dict[str, Any] = Field({}, description="Additional metadata")


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: List[Dict[str, str]]
    created_at: Optional[str] = None
    expires_at: Optional[str] = None


def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set")
    return api_key


def create_llm_instance(
    model: str,
    temperature: float,
    max_tokens: Optional[int],
) -> ChatOpenAI:
    """Create and return a ChatOpenAI instance with the given parameters."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(
    request: ChatCompletionRequest, api_key: str = Depends(get_openai_api_key)
):
    logger.info(f"\tReceived request: {request.json()}")
    try:
        # Initialize Redis client (only for conversation history)
        redis_client = get_redis_client()

        # Use configuration from the request
        model = request.model
        temperature = request.temperature
        max_tokens = request.max_tokens

        # Create the ChatOpenAI instance
        llm = create_llm_instance(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Use provided conversation_id or generate a new one
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Initialize messages list
        langchain_messages = []

        # If conversation_id is provided and exists in Redis, retrieve the history
        if request.conversation_id:
            stored_messages = get_conversation(redis_client, conversation_id)
            if stored_messages:
                # Convert stored messages to LangChain format
                langchain_messages = dict_to_langchain_messages(stored_messages)

        # Add new messages from the request
        for msg in request.messages:
            if msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))

        # Get the response
        response = llm.invoke(langchain_messages)

        # Add the assistant's response to the messages
        langchain_messages.append(AIMessage(content=response.content))

        # Save the updated conversation to Redis
        save_conversation(
            redis_client, conversation_id, langchain_to_dict_messages(langchain_messages)
        )

        # Extract token usage if available
        token_usage = None
        if hasattr(response, "response_metadata") and response.response_metadata:
            if "token_usage" in response.response_metadata:
                usage_data = response.response_metadata["token_usage"]
                token_usage = TokenUsage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )

        # Construct the response
        result = ChatCompletionResponse(
            content=response.content,
            conversation_id=conversation_id,
            usage=token_usage,
            additional_kwargs=response.additional_kwargs,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking OpenAI API: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "api": "LangChain OpenAI Wrapper"}


# Redis conversation utilities
def get_conversation_key(conversation_id: str) -> str:
    """Generate a Redis key for storing conversation history"""
    return f"conversation:{conversation_id}"


def save_conversation(client: redis.Redis, conversation_id: str, messages: List[Dict[str, str]]):
    """Save conversation history to Redis"""
    key = get_conversation_key(conversation_id)
    client.set(key, json.dumps(messages))
    client.expire(key, redis_config.conversation_ttl)


def get_conversation(client: redis.Redis, conversation_id: str) -> List[Dict[str, str]]:
    """Retrieve conversation history from Redis"""
    key = get_conversation_key(conversation_id)
    data = client.get(key)
    if data:
        return json.loads(data)
    return []


def langchain_to_dict_messages(messages) -> List[Dict[str, str]]:
    """Convert LangChain message objects to dictionaries for storage"""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


def dict_to_langchain_messages(messages):
    """Convert dictionary messages to LangChain message objects"""
    result = []
    for msg in messages:
        if msg["role"] == "system":
            result.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=msg["content"]))
    return result


@app.get("/v1/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation_history(
    conversation_id: str, api_key: str = Depends(get_openai_api_key)
):
    """Retrieve a specific conversation history"""
    try:
        redis_client = get_redis_client()
        messages = get_conversation(redis_client, conversation_id)

        if not messages:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

        # Get TTL information
        key = get_conversation_key(conversation_id)
        ttl = redis_client.ttl(key)

        if ttl > 0:
            expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        else:
            expires_at = None

        return ConversationResponse(
            conversation_id=conversation_id,
            messages=messages,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at,
        )
    except redis.RedisError as e:
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")


@app.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, api_key: str = Depends(get_openai_api_key)):
    """Delete a specific conversation history"""
    try:
        redis_client = get_redis_client()
        key = get_conversation_key(conversation_id)

        # Check if conversation exists
        if not redis_client.exists(key):
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

        # Delete the conversation
        redis_client.delete(key)

        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
    except redis.RedisError as e:
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")


@app.get("/v1/conversations", response_model=List[str])
async def list_conversations(api_key: str = Depends(get_openai_api_key)):
    """List all conversation IDs"""
    try:
        redis_client = get_redis_client()
        # Get all conversation keys and extract IDs
        keys = redis_client.keys("conversation:*")
        conversation_ids = [key.split(":", 1)[1] for key in keys]

        return conversation_ids
    except redis.RedisError as e:
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("langchain_server:app", host="0.0.0.0", port=8000, reload=True)
