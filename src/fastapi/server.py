from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Sample FastAPI Server",
    description="A simple API to learn FastAPI basics",
    version="0.1.0",
)


# Define data models
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
    is_offer: bool = False


# In-memory database
items_db = {
    1: Item(
        id=1,
        name="Laptop",
        description="Powerful development machine",
        price=999.99,
        is_offer=False,
    ),
    2: Item(
        id=2,
        name="Keyboard",
        description="Mechanical keyboard",
        price=99.99,
        is_offer=True,
    ),
    3: Item(id=3, name="Mouse", price=49.99, is_offer=False),
}


# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Sample FastAPI Server"}


# Get all items
@app.get("/items/", response_model=List[Item])
async def read_items():
    return list(items_db.values())


# Get a specific item by ID
@app.get("/items/{item_id}", response_model=Item)
async def read_item(
    item_id: int = Path(..., description="The ID of the item to retrieve")
):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]


# Create a new item
@app.post("/items/", response_model=Item, status_code=201)
async def create_item(item: Item):
    if item.id in items_db:
        raise HTTPException(
            status_code=400, detail=f"Item with ID {item.id} already exists"
        )
    items_db[item.id] = item
    return item


# Update an item
@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item: Item):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    if item_id != item.id:
        raise HTTPException(
            status_code=400, detail="Item ID in path does not match item ID in body"
        )
    items_db[item_id] = item
    return item


# Delete an item
@app.delete("/items/{item_id}", response_model=dict)
async def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_id]
    return {"message": f"Item {item_id} deleted successfully"}


# Search items by name
@app.get("/items/search/", response_model=List[Item])
async def search_items(
    query: str = Query(..., description="Search term to find items by name")
):
    matched_items = [
        item for item in items_db.values() if query.lower() in item.name.lower()
    ]
    return matched_items


# Run the server
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
