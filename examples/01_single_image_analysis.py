"""
Example: Single Image Analysis
Demonstrates basic image analysis workflow
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from main import LungCancerDetectionPipeline
from src.utils import setup_logging, set_seed
import logging

logger = logging.getLogger(__name__)


def main():
    """Run single image analysis example"""
    
    # Setup
    setup_logging(level="INFO")
    set_seed(config.system.random_seed)
    
    print("\n" + "=" * 80)
    print("LUNG CANCER DETECTION - SINGLE IMAGE ANALYSIS EXAMPLE")
    print("=" * 80)
    
    # Initialize pipeline
    print("\n[1/5] Initializing pipeline...")
    pipeline = LungCancerDetectionPipeline(config)
    
    # For this example, we'll create a dummy image
    # In real use, provide path to actual medical image
    import numpy as np
    dummy_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    
    # Create temporary image file
    import cv2
    temp_image_path = "/tmp/test_image.png"
    cv2.imwrite(temp_image_path, dummy_image)
    print(f"Created test image: {temp_image_path}")
    
    # Prepare patient information
    patient_info = {
        'patient_id': 'EXAMPLE_001',
        'name': 'John Doe',
        'age': 65,
        'gender': 'M',
        'smoker': True,
        'smoking_pack_years': 40,
        'family_history': True
    }
    
    print("\n[2/5] Analyzing image...")
    print(f"Patient: {patient_info['name']} (ID: {patient_info['patient_id']})")
    print(f"Age: {patient_info['age']}, Gender: {patient_info['gender']}")
    
    # Analyze
    results = pipeline.analyze_image(
        temp_image_path,
        patient_info=patient_info,
        generate_report=True
    )
    
    if results and results['status'] == 'success':
        print("\n[3/5] Analysis Results:")
        print("-" * 80)
        
        # Detection results
        det_result = results['detection_result']
        print(f"\nDetection Results:")
        print(f"  Nodules detected: {len(det_result.boxes)}")
        if len(det_result.confidences) > 0:
            print(f"  Average confidence: {det_result.confidences.mean():.3f}")
        print(f"  Processing time: {det_result.processing_time:.3f}s")
        
        # Classification results
        class_result = results['classification_result']
        if class_result:
            print(f"\nClassification Results:")
            print(f"  Predicted class: {class_result.class_name}")
            print(f"  Confidence: {class_result.confidence:.3f}")
            print(f"  Probabilities:")
            for class_name, prob in class_result.probabilities.items():
                print(f"    {class_name}: {prob:.3f}")
        
        # Risk assessment
        risk = results['risk_assessment']
        print(f"\nRisk Assessment:")
        print(f"  Risk Score: {risk.risk_score:.3f} (0-1 scale)")
        print(f"  Risk Level: {risk.risk_level.value}")
        print(f"  Model Confidence: {risk.confidence:.3f}")
        print(f"  Model Uncertainty: {risk.uncertainty:.3f}")
        print(f"  Recommendation: {risk.recommendation}")
        print(f"  Follow-up Period: {risk.followup_period}")
        
        if risk.key_features:
            print(f"  Key Features:")
            for feature in risk.key_features:
                print(f"    • {feature}")
        
        # Recommendations
        recs = results['recommendations']
        print(f"\nClinical Recommendations ({len(recs)} total):")
        for i, rec in enumerate(recs, 1):
            print(f"  [{i}] {rec.action}")
            print(f"      Priority: {rec.priority}")
            print(f"      Reason: {rec.reasoning}")
            print(f"      Follow-up: {rec.followup_days} days")
            if rec.additional_tests:
                print(f"      Tests: {', '.join(rec.additional_tests)}")
        
        # Report
        if results['report_path']:
            print(f"\n[4/5] Report Generation:")
            print(f"  Report saved: {results['report_path']}")
        
        print("\n[5/5] Analysis Complete!")
        print("=" * 80)
        
    else:
        print("\n❌ Analysis failed. Check logs for details.")


if __name__ == "__main__":
    main()
