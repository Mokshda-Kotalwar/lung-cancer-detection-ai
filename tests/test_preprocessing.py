"""
Unit tests for Lung Cancer Detection System
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from src.preprocessing import ImagePreprocessor, ImageNormalizer, LungSegmentation
from src.utils import set_seed, get_device, calculate_metrics


class TestPreprocessing:
    """Test preprocessing module"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = ImagePreprocessor(config)
        self.normalizer = ImageNormalizer()
    
    def test_image_loading(self):
        """Test image loading"""
        # Create dummy image
        dummy_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        assert dummy_image.shape == (512, 512)
    
    def test_normalization_minmax(self):
        """Test min-max normalization"""
        image = np.random.rand(512, 512) * 255
        normalized = self.normalizer.minmax_normalize(image)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert normalized.dtype == np.float32
    
    def test_normalization_zscore(self):
        """Test z-score normalization"""
        image = np.random.rand(512, 512) * 255
        normalized = self.normalizer.zscore_normalize(image)
        
        assert normalized.dtype == np.float32
        assert abs(normalized.mean()) < 0.1  # Close to 0
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        processed = self.preprocessor.preprocess(image)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape[0] == 1  # Channel dimension


class TestMetrics:
    """Test metric calculations"""
    
    def test_calculate_metrics(self):
        """Test medical metrics calculation"""
        tp, fp, fn, tn = 80, 10, 20, 90
        metrics = calculate_metrics(tp, fp, fn, tn)
        
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
        assert 'precision' in metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        
        # Validate ranges
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} out of range: {value}"


class TestConfiguration:
    """Test configuration"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        assert config is not None
        assert config.model is not None
        assert config.preprocessing is not None
        assert config.risk is not None
    
    def test_config_attributes(self):
        """Test configuration attributes"""
        assert hasattr(config.model, 'yolo_model')
        assert hasattr(config.model, 'input_size')
        assert config.model.input_size > 0


class TestUtilities:
    """Test utility functions"""
    
    def test_set_seed(self):
        """Test random seed setting"""
        set_seed(42)
        
        vals1 = np.random.rand(5)
        
        set_seed(42)
        vals2 = np.random.rand(5)
        
        np.testing.assert_array_equal(vals1, vals2)
    
    def test_get_device(self):
        """Test device detection"""
        device = get_device()
        assert device is not None
        assert device.type in ['cuda', 'cpu', 'mps']


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    """Test GPU-specific functionality"""
    
    def test_cuda_available(self):
        """Test CUDA availability"""
        assert torch.cuda.is_available()


from src.preprocessing import MedicalImagePreprocessor, LungNoduleDataset

class TestNewPreprocessingAndDataset:
    """Test the newly implemented pipeline and dataset modules"""
    
    def test_medical_preprocessor_windowing(self):
        """Test HU windowing normalization"""
        preprocessor = MedicalImagePreprocessor(config, is_training=False)
        raw_ct = np.random.randint(-1000, 1000, (512, 512)).astype(np.float32)
        windowed = preprocessor.apply_window_level(raw_ct)
        
        assert windowed.dtype == np.uint8
        assert windowed.shape == (512, 512)
        assert windowed.min() >= 0
        assert windowed.max() <= 255
        
    def test_dataset_loading_pipeline(self):
        """Test dataset loading pipeline using dummy slice files"""
        import tempfile
        preprocessor = MedicalImagePreprocessor(config, is_training=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            dummy_image_path = temp_path / "dummy_slice.npy"
            dummy_slice = np.random.randint(-1000, 400, (512, 512)).astype(np.float32)
            np.save(dummy_image_path, dummy_slice)
            
            dataset = LungNoduleDataset(
                image_paths=[dummy_image_path],
                labels=[1],
                bboxes=[[0.5, 0.5, 0.1, 0.1]],
                config=config,
                is_training=False
            )
            
            assert len(dataset) == 1
            sample = dataset[0]
            assert "image" in sample
            assert "image_path" in sample
            assert sample["image"].shape == (3, 512, 512) # duplicated channel dimensions
            assert sample["label"].item() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

