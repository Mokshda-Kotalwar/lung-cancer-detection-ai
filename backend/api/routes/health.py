from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def health_check():
    """Health check endpoint for Docker and Render"""
    return {"status": "ok", "service": "lung-cancer-detection-api"}
