import requests
import json
import argparse
from typing import List, Dict, Optional

BASE_URL = "http://localhost:8000"


def make_chat_request(
    prompt: str, model: str = "gpt-3.5-turbo", conversation_id: Optional[str] = None
) -> Dict:
    """
    Make a chat completion request to the LangChain server with Redis state support.

    Args:
        prompt: The user's message
        model: The OpenAI model to use
        conversation_id: Optional conversation ID to continue a previous conversation

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
        choices=["chat", "list", "get", "delete"],
        required=True,
        help="Action to perform: chat, list, get, or delete",
    )
    parser.add_argument("--prompt", help="Prompt message for chat")
    parser.add_argument("--conversation_id", help="Conversation ID for continuing a conversation")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")

    args = parser.parse_args()

    if args.action == "chat":
        if not args.prompt:
            print("Error: --prompt is required for chat action")
            return

        response = make_chat_request(args.prompt, args.model, args.conversation_id)

        if response:
            print("\nAI Response:")
            print(response["content"])
            print(f"\nConversation ID: {response['conversation_id']}")
            print("Use this ID to continue the conversation.")

    elif args.action == "list":
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
