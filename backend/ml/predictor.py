import os
import io
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from backend.core.config import settings
from config import config
from src.models.classifier import DenseNetClassifier
from src.risk import RiskScorer
from src.xai import GradCAM

logger = logging.getLogger(__name__)

class LungCancerPredictor:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 3
        self.class_names = ["Benign", "Malignant", "Uncertain"]

        resolved_model_path = self._resolve_model_path(model_path)
        self.model_path = resolved_model_path

        self.model = DenseNetClassifier(num_classes=self.num_classes, pretrained=False)

        if resolved_model_path:
            try:
                state_dict = torch.load(resolved_model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded model weights from {resolved_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model weights from {resolved_model_path}: {e}. Using random weights.")

        self.model.to(self.device)
        self.model.eval()

    def _resolve_model_path(self, model_path: str = None) -> str | None:
        candidates = []
        if model_path:
            candidates.append(model_path)

        env_path = os.getenv("MODEL_PATH")
        if env_path:
            candidates.append(env_path)

        if getattr(settings, "MODEL_PATH", None):
            candidates.append(settings.MODEL_PATH)

        project_root = Path(__file__).resolve().parents[2]
        candidates.extend([
            project_root / "models" / "checkpoints" / "test_densenet.pth",
            project_root / "models" / "checkpoints" / "best_densenet.pth",
            project_root / "models" / "checkpoints" / "best_model.pth",
            project_root / "models" / "checkpoints" / "model.pth",
        ])

        for candidate in candidates:
            if candidate is None:
                continue
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute():
                candidate_path = (project_root / candidate_path).resolve()
            if candidate_path.exists():
                return str(candidate_path)

        return None
        
        # Standard ImageNet transforms, as typically used for pretrained models
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.risk_scorer = RiskScorer(config)

    def _prepare_tensor(self, image_bytes: bytes) -> tuple:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image, tensor

    def predict(self, image_bytes: bytes, patient_info: dict = None) -> dict:
        try:
            image, tensor = self._prepare_tensor(image_bytes)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0)
                
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            probs_dict = {
                self.class_names[i]: float(probabilities[i].item())
                for i in range(self.num_classes)
            }
            
            # Default mock parameters for detection info since YOLO is not running in backend currently
            detection_count = 1 if predicted_idx.item() == 1 else 0
            
            # Calculate Risk if patient_info provided
            risk_info = {}
            if patient_info:
                clinical_features = {}
                if "age" in patient_info and patient_info["age"]:
                    clinical_features["age"] = patient_info["age"] / 100.0
                if "smoker" in patient_info and patient_info["smoker"]:
                    clinical_features["smoking_pack_years"] = 0.8
                
                risk_assessment = self.risk_scorer.calculate_risk(
                    classification_confidence=float(confidence.item()),
                    detection_count=detection_count,
                    detection_size=12.5 if detection_count > 0 else 0.0,
                    detection_confidence=float(confidence.item()),
                    clinical_features=clinical_features
                )
                risk_info = {
                    "risk_score": risk_assessment.risk_score,
                    "risk_level": risk_assessment.risk_level.value,
                    "recommendation": risk_assessment.recommendation
                }
            
            return {
                "prediction": self.class_names[predicted_idx.item()],
                "confidence": float(confidence.item()),
                "probabilities": probs_dict,
                **risk_info
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e

    def generate_gradcam(self, image_bytes: bytes) -> bytes:
        try:
            image, tensor = self._prepare_tensor(image_bytes)
            # Use original image as numpy array for GradCAM overlay
            original_image = np.array(image)
            
            # Predict to get target class
            
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0)
            _, predicted_idx = torch.max(probabilities, 0)
            
            target_layer = "backbone.features.norm5"
            gradcam = GradCAM(self.model, target_layer=target_layer, device=self.device)
            
            heatmap_overlay = gradcam.visualize(
                input_tensor=tensor,
                original_image=original_image,
                target_class=predicted_idx.item(),
                colormap="jet",
                alpha=0.4
            )
            
            # Convert back to bytes
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
            if not is_success:
                raise ValueError("Could not encode Grad-CAM image to PNG format")
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"GradCAM generation failed: {e}")
            raise e

# Instantiate a global predictor
predictor = LungCancerPredictor()
