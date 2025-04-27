import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

app = FastAPI(
    title="LangChain OpenAI API Wrapper",
    description="A FastAPI server that wraps the LangChain OpenAI chat completion API",
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


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    content: str = Field(..., description="The generated text from the model")
    model: str = Field(..., description="The model used for generation")
    usage: Optional[TokenUsage] = Field(None, description="Token usage information")
    additional_kwargs: Dict[str, Any] = Field({}, description="Additional metadata")


def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set")
    return api_key


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(
    request: ChatCompletionRequest, api_key: str = Depends(get_openai_api_key)
):
    try:
        # Initialize the ChatOpenAI instance
        llm = ChatOpenAI(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            api_key=api_key,
        )

        # Convert messages to LangChain format
        langchain_messages = []
        for msg in request.messages:
            if msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))

        # Get the response
        response = llm.invoke(langchain_messages)

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
            model=llm.model_name,
            usage=token_usage,
            additional_kwargs=response.additional_kwargs,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking OpenAI API: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "api": "LangChain OpenAI Wrapper"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("langchain_server:app", host="0.0.0.0", port=8000, reload=True)
