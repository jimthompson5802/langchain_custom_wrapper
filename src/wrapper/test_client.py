import requests
import json
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class AIMessage(Message):
    """Message from an AI."""

    response_metadata: Optional[Dict[str, Any]] = None
    role: str = "assistant"
    additional_kwargs: Optional[Dict[str, Any]] = None


class HumanMessage(Message):
    """Message from a human."""

    role: str = "user"


class SystemMessage(Message):
    """Message from the system."""

    role: str = "system"


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class FastAPIChatOpenAI:
    """
    A class that mimics the LangChain OpenAI chat completion class.
    It communicates with the FastAPI server to get completions.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers = {"Content-Type": "application/json"}

        # Check health of the API server
        self._check_health()

    def _check_health(self) -> Dict[str, Any]:
        """Check if the API server is healthy."""
        try:
            health_response = requests.get(f"{self.base_url}/health")
            health_response.raise_for_status()
            return health_response.json()
        except requests.exceptions.RequestException as e:
            print(f"Warning: API server health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def _convert_messages_to_api_format(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert LangChain style messages to API format."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def invoke(self, messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) -> AIMessage:
        """
        Invoke the chat model to get a completion.

        Args:
            messages: List of LangChain style messages

        Returns:
            AIMessage: The response from the model
        """
        chat_url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "messages": self._convert_messages_to_api_format(messages),
            "model": self.model,
            "temperature": self.temperature,
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        try:
            response = requests.post(chat_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()

            # Create an AIMessage from the response
            ai_message = AIMessage(content=result["content"])

            # Add response metadata as attributes
            # ai_message.model = result["model"]

            # Add token usage if available
            if "usage" in result and result["usage"]:
                ai_message.response_metadata = {"token_usage": result["usage"]}

            # Add any additional kwargs
            ai_message.additional_kwargs = result.get("additional_kwargs", {})

            return ai_message

        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            if hasattr(e, "response") and e.response:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            raise e


def test_langchain_api():
    """
    Test the LangChain OpenAI wrapper API with a sample request.
    """
    # Create an instance of the FastAPIChatOpenAI class
    chat = FastAPIChatOpenAI()

    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of Hawaii?"),
    ]

    try:
        # Invoke the model
        response = chat.invoke(messages)

        print("\n--- Response from LangChain OpenAI API ---")
        print(f"Content: {response.content}")

        if hasattr(response, "response_metadata") and response.response_metadata:
            if "token_usage" in response.response_metadata:
                print("\n--- Token Usage ---")
                for key, value in response.response_metadata["token_usage"].items():
                    print(f"{key}: {value}")

        if hasattr(response, "model"):
            print(f"\nModel used: {response.model}")

        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            print("\n--- Additional Metadata ---")
            for key, value in response.additional_kwargs.items():
                print(f"{key}: {value}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    # if not os.environ.get("OPENAI_API_KEY"):
    #     print("Warning: OPENAI_API_KEY environment variable not set.")
    #     print("The server will not be able to process requests without it.")

    test_langchain_api()
