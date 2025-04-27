import requests
import json


class FastAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def get_welcome_message(self):
        """Get the welcome message from the root endpoint"""
        response = requests.get(f"{self.base_url}/")
        return response.json()

    def get_all_items(self):
        """Get all items from the server"""
        response = requests.get(f"{self.base_url}/items/")
        return response.json()

    def get_item_by_id(self, item_id):
        """Get a specific item by ID"""
        response = requests.get(f"{self.base_url}/items/{item_id}")
        if response.status_code == 404:
            return {"error": "Item not found"}
        return response.json()

    def search_items(self, query):
        """Search for items by name"""
        response = requests.get(
            f"{self.base_url}/items/search/", params={"query": query}
        )
        return response.json()

    def create_item(self, item_data):
        """Create a new item"""
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{self.base_url}/items/", data=json.dumps(item_data), headers=headers
        )
        return response.json(), response.status_code

    def update_item(self, item_id, item_data):
        """Update an existing item"""
        headers = {"Content-Type": "application/json"}
        response = requests.put(
            f"{self.base_url}/items/{item_id}",
            data=json.dumps(item_data),
            headers=headers,
        )
        if response.status_code == 404:
            return {"error": "Item not found"}, response.status_code
        return response.json(), response.status_code

    def delete_item(self, item_id):
        """Delete an item"""
        response = requests.delete(f"{self.base_url}/items/{item_id}")
        if response.status_code == 404:
            return {"error": "Item not found"}, response.status_code
        return response.json(), response.status_code


def demo_client():
    """Run a demonstration of the client"""
    client = FastAPIClient()

    # Print welcome message
    print("\n1. Getting welcome message:")
    print(client.get_welcome_message())

    # Get all items
    print("\n2. Getting all items:")
    print(json.dumps(client.get_all_items(), indent=2))

    # Get item by ID
    print("\n3. Getting item with ID 2:")
    print(json.dumps(client.get_item_by_id(2), indent=2))

    # Search for items
    print("\n4. Searching for items with 'key' in the name:")
    print(json.dumps(client.search_items("key"), indent=2))

    # Create a new item
    new_item = {
        "id": 4,
        "name": "Monitor",
        "description": "27-inch 4K display",
        "price": 349.99,
        "is_offer": True,
    }
    print("\n5. Creating a new item:")
    result, status_code = client.create_item(new_item)
    print(f"Status: {status_code}")
    print(json.dumps(result, indent=2))

    # Update an item
    updated_item = {
        "id": 3,
        "name": "Wireless Mouse",
        "description": "Ergonomic wireless mouse",
        "price": 59.99,
        "is_offer": True,
    }
    print("\n6. Updating item with ID 3:")
    result, status_code = client.update_item(3, updated_item)
    print(f"Status: {status_code}")
    print(json.dumps(result, indent=2))

    # Get all items after update
    print("\n7. Getting all items after update:")
    print(json.dumps(client.get_all_items(), indent=2))

    # Delete an item
    print("\n8. Deleting item with ID 2:")
    result, status_code = client.delete_item(2)
    print(f"Status: {status_code}")
    print(json.dumps(result, indent=2))

    # Get all items after deletion
    print("\n9. Getting all items after deletion:")
    print(json.dumps(client.get_all_items(), indent=2))


if __name__ == "__main__":
    print("FastAPI Client Demo")
    print("===================")
    print("Note: Make sure the server is running before executing this client.")

    choice = input("Do you want to run the demo? (yes/no): ").lower()
    if choice == "yes" or choice == "y":
        demo_client()
    else:
        print("Demo cancelled. You can import FastAPIClient in your own scripts.")
