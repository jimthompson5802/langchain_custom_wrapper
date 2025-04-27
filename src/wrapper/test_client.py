from typing import Dict, Any, List
from src.wrapper.fastapi_chat import FastAPIChatOpenAI, SystemMessage, HumanMessage, AIMessage


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
