import pytest
import torch
from src.xai import GradCAM
from src.models.classifier import DenseNetClassifier

@pytest.fixture
def mock_model():
    # Load a lightweight model instance for testing GradCAM hooking mechanism
    model = DenseNetClassifier(num_classes=3, pretrained=False)
    return model

def test_gradcam_initialization(mock_model):
    """Test that GradCAM can properly attach to the target layer"""
    try:
        # For DenseNet, backbone.features.norm5 is the target layer
        gradcam = GradCAM(mock_model, target_layer="backbone.features.norm5", device="cpu")
        assert gradcam is not None
    except Exception as e:
        pytest.fail(f"GradCAM initialization failed: {e}")

def test_gradcam_heatmap_generation(mock_model):
    """Test that GradCAM generates a valid heatmap tensor given a mock input"""
    gradcam = GradCAM(mock_model, target_layer="backbone.features.norm5", device="cpu")
    
    # Create a dummy image tensor (1 batch, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Test heatmap generation for class 0
    heatmap = gradcam.generate_heatmap(dummy_input, target_class=0)
    
    # Assert heatmap is not None and has the expected spatial dimensions
    assert heatmap is not None
    assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"
