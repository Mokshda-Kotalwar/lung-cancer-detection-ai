from fastapi import APIRouter, Depends
from typing import List
from backend.api.deps import get_current_user, get_db

router = APIRouter()

@router.get("/", response_model=List[dict])
async def get_history(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    history_cursor = db["history"].find({"user_id": current_user["id"]})
    history_list = await history_cursor.to_list(length=100)
    
    # Convert ObjectId to string for JSON serialization
    for doc in history_list:
        doc["_id"] = str(doc["_id"])
        
    return history_list
