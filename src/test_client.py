import requests
import json
import os


def test_langchain_api():
    """
    Test the LangChain OpenAI wrapper API with a sample request.
    """
    base_url = "http://localhost:8000"

    # Test the health endpoint
    health_response = requests.get(f"{base_url}/health")
    print(f"Health check status: {health_response.status_code}")
    print(f"Health check response: {health_response.json()}")

    # Test the chat completions endpoint
    chat_url = f"{base_url}/v1/chat/completions"

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Hawaii?"},
        ],
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(chat_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses

        result = response.json()

        print("\n--- Response from LangChain OpenAI API ---")
        print(f"Content: {result['content']}")

        if "usage" in result and result["usage"]:
            print("\n--- Token Usage ---")
            for key, value in result["usage"].items():
                print(f"{key}: {value}")

        print(f"\nModel used: {result['model']}")

        if "additional_kwargs" in result and result["additional_kwargs"]:
            print("\n--- Additional Metadata ---")
            for key, value in result["additional_kwargs"].items():
                print(f"{key}: {value}")

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, "response") and e.response:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("The server will not be able to process requests without it.")

    test_langchain_api()
