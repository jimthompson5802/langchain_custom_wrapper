import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Check if OpenAI API key is available
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key as an environment variable.")
else:
    print(f"OPENAI_API_KEY found: {api_key[:5]}...")  # Show only first few characters for security

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # Create a prompt
    prompt = "What is the capital of Hawaii?"
    messages = [HumanMessage(content=prompt)]

    # Get the response
    print(f"\nSending prompt: '{prompt}'")
    response = llm.invoke(messages)

    # Display the response with metadata
    print("\n--- Response ---")
    print(f"Content: {response.content}")
    print("\n--- Metadata ---")
    for key, value in response.additional_kwargs.items():
        print(f"{key}: {value}")

    # If you want to access token usage info (if available)
    if hasattr(response, "response_metadata") and response.response_metadata:
        print("\n--- Usage Statistics ---")
        if "token_usage" in response.response_metadata:
            token_usage = response.response_metadata["token_usage"]
            for k, v in token_usage.items():
                print(f"{k}: {v}")

    print("\n--- Model Info ---")
    print(f"Model: {llm.model_name}")
    print(f"Temperature: {llm.temperature}")
