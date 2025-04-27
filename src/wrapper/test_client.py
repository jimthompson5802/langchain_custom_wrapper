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
        self.conversation_id = None  # Track conversation ID for stateful chat
        self.model_id = None  # Track model ID for cached model configuration

        # Check health of the API server
        self._check_health()

        # Create model configuration in Redis
        self._create_model()

    def _check_health(self) -> Dict[str, Any]:
        """Check if the API server is healthy."""
        try:
            health_response = requests.get(f"{self.base_url}/health")
            health_response.raise_for_status()
            return health_response.json()
        except requests.exceptions.RequestException as e:
            print(f"Warning: API server health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def _create_model(self) -> None:
        """Create a model configuration in Redis."""
        try:
            model_url = f"{self.base_url}/v1/models/create"

            payload = {
                "model": self.model,
                "temperature": self.temperature,
            }

            if self.max_tokens:
                payload["max_tokens"] = self.max_tokens

            response = requests.post(model_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()
            self.model_id = result["model_id"]
            print(f"Created model configuration with ID: {self.model_id}")

        except requests.exceptions.RequestException as e:
            print(f"Error creating model configuration: {e}")
            if hasattr(e, "response") and e.response:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")

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

        # Add model_id for the cached model configuration
        if self.model_id:
            payload["model_id"] = self.model_id

        # Add conversation_id if we have one to maintain state
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id

        try:
            response = requests.post(chat_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()

            # Store the conversation_id for future interactions
            if "conversation_id" in result:
                self.conversation_id = result["conversation_id"]

            # Create an AIMessage from the response
            ai_message = AIMessage(content=result["content"])

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

    def get_conversation_history(self) -> Dict[str, Any]:
        """
        Retrieve the current conversation history.

        Returns:
            Dictionary containing the conversation details
        """
        if not self.conversation_id:
            return {"error": "No active conversation"}

        try:
            url = f"{self.base_url}/v1/conversations/{self.conversation_id}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving conversation: {e}")
            return {"error": str(e)}

    def list_conversations(self) -> List[str]:
        """
        List all available conversation IDs.

        Returns:
            List of conversation IDs
        """
        try:
            url = f"{self.base_url}/v1/conversations"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error listing conversations: {e}")
            return []

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available model configurations.

        Returns:
            List of model configurations
        """
        try:
            url = f"{self.base_url}/v1/models"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []

    def delete_conversation(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete a conversation.

        Args:
            conversation_id: ID of conversation to delete. If None, uses the current conversation.

        Returns:
            Response from the server
        """
        conv_id = conversation_id or self.conversation_id
        if not conv_id:
            return {"error": "No conversation ID provided"}

        try:
            url = f"{self.base_url}/v1/conversations/{conv_id}"
            response = requests.delete(url)
            response.raise_for_status()

            # Reset conversation_id if we deleted the current conversation
            if conv_id == self.conversation_id:
                self.conversation_id = None

            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error deleting conversation: {e}")
            return {"error": str(e)}


def test_langchain_api():
    """
    Test the LangChain OpenAI wrapper API with a sample request.
    """
    # Create an instance of the FastAPIChatOpenAI class
    # This will also create a model configuration in Redis
    print("\n=== Creating ChatOpenAI instance and model configuration ===")
    chat = FastAPIChatOpenAI(model="gpt-4o-mini")

    print(f"Model ID: {chat.model_id}")

    print("\n=== Testing Stateful Conversation with Redis ===\n")

    # First interaction: Ask about Hawaii
    print("First interaction: Asking about Hawaii")
    messages1 = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of Hawaii?"),
    ]

    try:
        # Invoke the model
        response1 = chat.invoke(messages1)

        print("\n--- Response 1 ---")
        print(f"Content: {response1.content}")
        print(f"Conversation ID: {chat.conversation_id}")

        # Display token usage if available
        if hasattr(response1, "response_metadata") and response1.response_metadata:
            if "token_usage" in response1.response_metadata:
                print("\n--- Token Usage ---")
                for key, value in response1.response_metadata["token_usage"].items():
                    print(f"{key}: {value}")

        # Second interaction: Follow-up question using stored conversation state
        print("\n\nSecond interaction: Follow-up question")
        # Note: We only need to send the new message, the history is stored in Redis
        messages2 = [HumanMessage(content="What's another interesting fact about Hawaii?")]

        # The conversation ID is automatically included from the previous interaction
        # The model ID is automatically included as well
        response2 = chat.invoke(messages2)

        print("\n--- Response 2 ---")
        print(f"Content: {response2.content}")
        print(f"Using conversation ID: {chat.conversation_id}")
        print(f"Using model ID: {chat.model_id}")

        # Display token usage for second response if available
        if hasattr(response2, "response_metadata") and response2.response_metadata:
            if "token_usage" in response2.response_metadata:
                print("\n--- Token Usage ---")
                for key, value in response2.response_metadata["token_usage"].items():
                    print(f"{key}: {value}")

        # Get conversation history
        print("\n\nRetrieving full conversation history from Redis:")
        conversation = chat.get_conversation_history()
        if "messages" in conversation:
            for i, msg in enumerate(conversation["messages"]):
                role = msg["role"].upper()
                content = msg["content"]
                print(f"\n[{i+1}. {role}]: {content}")

        # List all conversations
        print("\n\nListing all available conversations:")
        conversations = chat.list_conversations()
        for conv_id in conversations:
            if conv_id == chat.conversation_id:
                print(f"- {conv_id} (current)")
            else:
                print(f"- {conv_id}")

        # List all model configurations
        print("\n\nListing all available model configurations:")
        models = chat.list_models()
        for model in models:
            model_id = model.get("model_id", "unknown")
            model_config = model.get("config", {})
            model_name = model_config.get("model", "unknown")
            temp = model_config.get("temperature", "unknown")

            if model_id == chat.model_id:
                print(f"- {model_id} (current): {model_name}, temp={temp}")
            else:
                print(f"- {model_id}: {model_name}, temp={temp}")

        # Optionally clean up by deleting the conversation
        delete_choice = input("\nDelete this conversation? (y/n): ").lower()
        if delete_choice == "y" or delete_choice == "yes":
            deletion_result = chat.delete_conversation()
            print(f"Deletion result: {deletion_result}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    # if not os.environ.get("OPENAI_API_KEY"):
    #     print("Warning: OPENAI_API_KEY environment variable not set.")
    #     print("The server will not be able to process requests without it.")

    test_langchain_api()
