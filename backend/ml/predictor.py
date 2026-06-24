import torch
from torchvision import transforms
from PIL import Image
import io
import logging
from src.models.classifier import DenseNetClassifier

logger = logging.getLogger(__name__)

class LungCancerPredictor:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 3
        self.class_names = ["Benign", "Malignant", "Uncertain"]
        
        self.model = DenseNetClassifier(num_classes=self.num_classes, pretrained=False)
        
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model weights from {model_path}: {e}. Using random weights.")
                
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet transforms, as typically used for pretrained models
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0)
                
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            probs_dict = {
                self.class_names[i]: float(probabilities[i].item())
                for i in range(self.num_classes)
            }
            
            return {
                "prediction": self.class_names[predicted_idx.item()],
                "confidence": float(confidence.item()),
                "probabilities": probs_dict
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e

# Instantiate a global predictor, we can defer loading the path to initialization or config
predictor = LungCancerPredictor()
