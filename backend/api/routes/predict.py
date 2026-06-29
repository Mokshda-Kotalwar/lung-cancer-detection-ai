from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
from typing import Optional
import json
from backend.api.deps import get_current_active_user, get_db
from backend.models.schemas import PredictionResult, ConfidenceResult, PatientDetails
from backend.ml.predictor import predictor
from src.reporting import ReportGenerator
from config import config
import os
import tempfile
from fastapi.responses import FileResponse
import numpy as np
import cv2


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

class MockDetection:
    def __init__(self):
        self.boxes = []
        self.confidences = np.array([])

class MockClassification:
    def __init__(self, class_name, confidence):
        self.class_name = class_name
        self.confidence = confidence

class MockRiskLevel:
    def __init__(self, level):
        self.value = level

class MockRiskAssessment:
    def __init__(self, risk_score, risk_level_str, confidence):
        self.risk_score = risk_score
        self.risk_level = MockRiskLevel(risk_level_str)
        self.confidence = confidence
        self.uncertainty = 0.05
        self.key_features = ["Age factor", "Smoking history"]

class MockRecommendation:
    def __init__(self, recommendation_str):
        self.action = recommendation_str
        self.priority = "High"
        self.reasoning = "Based on model prediction"
        self.followup_days = 30
        self.additional_tests = ["Biopsy consultation"]

@router.post("/report")
async def generate_report(
    file: UploadFile = File(...),
    patient_details_json: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_active_user)
):
    """Endpoint to generate and return a PDF report"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        image_bytes = await file.read()
        
        # Predict using ML model
        patient_info = json.loads(patient_details_json) if patient_details_json else {}
        result = predictor.predict(image_bytes, patient_info=patient_info)
        
        # Generate GradCAM
        gradcam_bytes = predictor.generate_gradcam(image_bytes)
        
        # Setup mocks for ReportGenerator
        det_result = MockDetection()
        cls_result = MockClassification(result["prediction"], result["confidence"])
        risk_result = MockRiskAssessment(
            result.get("risk_score", 0.5), 
            result.get("risk_level", "Unknown"), 
            result["confidence"]
        )
        recs = [MockRecommendation(result.get("recommendation", "Consult physician"))]
        
        # Decode images for ReportLab
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        nparr_g = np.frombuffer(gradcam_bytes, np.uint8)
        gradcam_image = cv2.imdecode(nparr_g, cv2.IMREAD_COLOR)
        
        # Generate PDF
        generator = ReportGenerator(config)
        fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        
        success = generator.generate_pdf_report(
            output_path=temp_path,
            patient_info=patient_info,
            detection_result=det_result,
            classification_result=cls_result,
            risk_assessment=risk_result,
            recommendations=recs,
            gradcam_image=gradcam_image,
            original_image=original_image
        )
        
        if not success:
            raise Exception("Failed to generate PDF report")
            
        return FileResponse(
            temp_path, 
            media_type='application/pdf', 
            filename=f"lung_cancer_report_{patient_info.get('patient_id', 'unknown')}.pdf",
            background=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

