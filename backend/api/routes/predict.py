from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from backend.api.deps import get_current_user, get_db
from backend.models.schemas import PredictionResult
from backend.ml.predictor import predictor

router = APIRouter()

@router.post("/", response_model=PredictionResult)
async def predict_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        # Read the file
        image_bytes = await file.read()
        
        # Predict using the ML model
        result = predictor.predict(image_bytes)
        
        # Save prediction history in MongoDB
        history_record = {
            "user_id": current_user["id"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "filename": file.filename,
            "content_type": file.content_type
        }
        
        await db["history"].insert_one(history_record)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
