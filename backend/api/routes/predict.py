from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
from typing import Optional
import json
from backend.api.deps import get_current_active_user, get_db
from backend.models.schemas import PredictionResult, ConfidenceResult, PatientDetails
from backend.ml.predictor import predictor

router = APIRouter()

@router.post("/", response_model=PredictionResult)
async def predict_image(
    file: UploadFile = File(...),
    patient_details_json: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_active_user),
    db = Depends(get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        # Read the file
        image_bytes = await file.read()
        
        # Parse patient details
        patient_info = None
        if patient_details_json:
            patient_info = json.loads(patient_details_json)
        
        # Predict using the ML model
        result = predictor.predict(image_bytes, patient_info=patient_info)
        
        # Save prediction history in MongoDB
        history_record = {
            "user_id": current_user["id"],
            "patient_details": patient_info or {},
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "risk_score": result.get("risk_score"),
            "risk_level": result.get("risk_level"),
            "recommendation": result.get("recommendation"),
            "filename": file.filename,
            "content_type": file.content_type
        }
        
        await db["history"].insert_one(history_record)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/confidence", response_model=ConfidenceResult)
async def get_confidence(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_active_user)
):
    """Endpoint specifically for returning only confidence score (Phase 4 requirement)"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        image_bytes = await file.read()
        result = predictor.predict(image_bytes)
        
        return ConfidenceResult(
            prediction=result["prediction"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/gradcam")
async def get_gradcam(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_active_user)
):
    """Endpoint to return Grad-CAM visualization map"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        image_bytes = await file.read()
        gradcam_bytes = predictor.generate_gradcam(image_bytes)
        
        return Response(content=gradcam_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
