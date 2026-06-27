from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from backend.api.deps import get_current_active_user, get_db

router = APIRouter()

@router.get("/", response_model=List[dict])
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    search_query: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user),
    db = Depends(get_db)
):
    query = {"user_id": current_user["id"]}
    if search_query:
        # Simple text search on patient ID or name
        query["$or"] = [
            {"patient_details.patient_id": {"$regex": search_query, "$options": "i"}},
            {"patient_details.name": {"$regex": search_query, "$options": "i"}}
        ]
        
    history_cursor = db["history"].find(query).sort("_id", -1)
    history_list = await history_cursor.to_list(length=limit)
    
    # Convert ObjectId to string for JSON serialization
    for doc in history_list:
        doc["_id"] = str(doc["_id"])
        
    return history_list
