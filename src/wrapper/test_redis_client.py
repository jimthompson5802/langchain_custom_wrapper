import requests
import argparse
from typing import List, Dict, Optional

BASE_URL = "http://localhost:8000"


def create_model(model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> Dict:
    """
    Create a model configuration and store it in Redis.

    Args:
        model: The OpenAI model to use
        temperature: The temperature parameter for the model

    Returns:
        The server's response as a dictionary with the model_id
    """
    url = f"{BASE_URL}/v1/models/create"

    payload = {
        "model": model,
        "temperature": temperature,
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}


def make_chat_request(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    conversation_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict:
    """
    Make a chat completion request to the LangChain server with Redis state support.

    Args:
        prompt: The user's message
        model: The OpenAI model to use (only used if model_id is not provided)
        conversation_id: Optional conversation ID to continue a previous conversation
        model_id: Optional model ID to use a previously created model configuration

    Returns:
        The server's response as a dictionary
    """
    url = f"{BASE_URL}/v1/chat/completions"

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "temperature": 0.7,
    }

    # Add conversation_id if provided to continue the conversation
    if conversation_id:
        payload["conversation_id"] = conversation_id

    # Add model_id if provided to use the cached model configuration
    if model_id:
        payload["model_id"] = model_id

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}


def get_conversation(conversation_id: str) -> Dict:
    """
    Retrieve a conversation from the server.

    Args:
        conversation_id: The ID of the conversation to retrieve

    Returns:
        The conversation data as a dictionary
    """
    url = f"{BASE_URL}/v1/conversations/{conversation_id}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}


def list_conversations() -> List[str]:
    """
    List all conversation IDs stored in Redis.

    Returns:
        A list of conversation IDs
    """
    url = f"{BASE_URL}/v1/conversations"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return []


def list_models() -> List[Dict]:
    """
    List all model configurations stored in Redis.

    Returns:
        A list of model configurations
    """
    url = f"{BASE_URL}/v1/models"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return []


def delete_conversation(conversation_id: str) -> Dict:
    """
    Delete a conversation from Redis.

    Args:
        conversation_id: The ID of the conversation to delete

    Returns:
        The server's response as a dictionary
    """
    url = f"{BASE_URL}/v1/conversations/{conversation_id}"

    response = requests.delete(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Test client for LangChain server with Redis state"
    )
    parser.add_argument(
        "--action",
        choices=["create-model", "chat", "list-models", "list-conversations", "get", "delete"],
        required=True,
        help="Action to perform: create-model, chat, list-models, list-conversations, get, or delete",  # NOQA: E501
    )
    parser.add_argument("--prompt", help="Prompt message for chat")
    parser.add_argument("--conversation_id", help="Conversation ID for continuing a conversation")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--model_id", help="Model ID to use a cached model configuration")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for model generation"
    )

    args = parser.parse_args()

    if args.action == "create-model":
        response = create_model(args.model, args.temperature)

        if response:
            print("\nModel Configuration Created:")
            print(f"Model ID: {response['model_id']}")
            print(f"Model: {response['model']}")
            print(f"Status: {response['status']}")
            print(f"Message: {response['message']}")
            print("\nUse this Model ID with the --model_id parameter to reuse this configuration.")

    elif args.action == "chat":
        if not args.prompt:
            print("Error: --prompt is required for chat action")
            return

        response = make_chat_request(args.prompt, args.model, args.conversation_id, args.model_id)

        if response:
            print("\nAI Response:")
            print(response["content"])
            print(f"\nConversation ID: {response['conversation_id']}")
            if args.model_id:
                print(f"Using model ID: {args.model_id}")
            print("Use this Conversation ID to continue the conversation.")

    elif args.action == "list-models":
        models = list_models()

        if models:
            print("\nAvailable Model Configurations:")
            for model in models:
                model_id = model.get("model_id", "unknown")
                config = model.get("config", {})
                model_name = config.get("model", "unknown")
                temp = config.get("temperature", "unknown")
                print(f"- {model_id}: {model_name}, temperature={temp}")
        else:
            print("No model configurations found.")

    elif args.action == "list-conversations":
        conversation_ids = list_conversations()

        if conversation_ids:
            print("\nAvailable Conversations:")
            for conv_id in conversation_ids:
                print(f"- {conv_id}")
        else:
            print("No conversations found.")

    elif args.action == "get":
        if not args.conversation_id:
            print("Error: --conversation_id is required for get action")
            return

        conversation = get_conversation(args.conversation_id)

        if conversation:
            print(f"\nConversation: {conversation['conversation_id']}")
            print(f"Created: {conversation.get('created_at', 'unknown')}")
            print(f"Expires: {conversation.get('expires_at', 'unknown')}")
            print("\nMessages:")

            for msg in conversation["messages"]:
                role = msg["role"].upper()
                content = msg["content"]
                print(f"\n[{role}]: {content}")

    elif args.action == "delete":
        if not args.conversation_id:
            print("Error: --conversation_id is required for delete action")
            return

        response = delete_conversation(args.conversation_id)

        if response:
            print(response["message"])


if __name__ == "__main__":
    main()
