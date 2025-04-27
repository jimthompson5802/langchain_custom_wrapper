from src.wrapper.fastapi_chat import FastAPIChatOpenAI, HumanMessage
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run custom LLM testbed with specified model.")
parser.add_argument(
    "--model", type=str, default="gpt-3.5-turbo", help="Model to use (default: gpt-3.5-turbo)"
)
args = parser.parse_args()

# Initialize the LLM
llm = FastAPIChatOpenAI(
    model=args.model,
    temperature=0.7,
)

# Create a promptcl
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
