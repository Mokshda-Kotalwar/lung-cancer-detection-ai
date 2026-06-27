"""
Main entry point for Lung Cancer Detection System
Provides CLI interface for image analysis
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

from yaml import parser

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config, OUTPUTS_DIR
from src.utils import setup_logging, set_seed, get_device
from src.preprocessing import ImagePreprocessor
from src.detection import EnsembleDetector
from src.xai import GradCAM
from src.risk import RiskScorer, RecommendationEngine
from src.reporting import ReportGenerator

logger = logging.getLogger(__name__)


class LungCancerDetectionPipeline:
    """Complete analysis pipeline"""
    
    def __init__(self, config_obj):
        """Initialize pipeline"""
        self.config = config_obj
        self.preprocessor = ImagePreprocessor(config_obj)
        self.ensemble = EnsembleDetector(config_obj)
        self.risk_scorer = RiskScorer(config_obj)
        self.recommendation_engine = RecommendationEngine(config_obj)
        self.report_generator = ReportGenerator(config_obj)
        
        logger.info("Pipeline initialized")
    
    def analyze_image(self, image_path: str, patient_info: dict = None, 
                     generate_report: bool = True) -> dict:
        """
        Complete image analysis pipeline
        
        Args:
            image_path: Path to medical image
            patient_info: Optional patient information
            generate_report: Whether to generate PDF report
        
        Returns:
            Analysis results dictionary
        """
        try:
            # Load image
            logger.info(f"Loading image: {image_path}")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Preprocess
            logger.info("Preprocessing image...")
            processed_image = self.preprocessor.preprocess(image)
            
            # Detection and classification
            logger.info("Running detection and classification...")
            detection_result, classification_result = self.ensemble.process(image)
            
            # Risk assessment
            logger.info("Calculating risk assessment...")
            risk_assessment = self.risk_scorer.calculate_risk(
                classification_confidence=classification_result.confidence if classification_result else 0.5,
                detection_count=len(detection_result.boxes) if detection_result else 0,
                detection_size=10.0,  # Placeholder
                detection_confidence=detection_result.confidences.mean() if detection_result and len(detection_result.confidences) > 0 else 0
            )
            
            # Generate recommendations
            logger.info("Generating recommendations...")
            recommendations = self.recommendation_engine.generate_recommendations(
                risk_assessment,
                patient_info=patient_info
            )
            
            # Generate report
            if generate_report:
                logger.info("Generating report...")
                report_path = OUTPUTS_DIR / "reports" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.report_generator.generate_pdf_report(
                    str(report_path),
                    patient_info=patient_info or {},
                    detection_result=detection_result,
                    classification_result=classification_result,
                    risk_assessment=risk_assessment,
                    recommendations=recommendations,
                    original_image=image
                )
            
            # Compile results
            results = {
                'status': 'success',
                'image_path': image_path,
                'detection_result': detection_result,
                'classification_result': classification_result,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'report_path': report_path if generate_report else None
            }
            
            logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Lung Cancer Detection AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python main.py --image path/to/image.png
  
  # Analyze with patient info
  python main.py --image path/to/image.png --patient-id P001 --name "John Doe"
  
  # Run Streamlit web app
  python main.py --web
        """
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to medical image for analysis"
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="UNKNOWN",
        help="Patient ID"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Unknown",
        help="Patient name"
    )
    parser.add_argument(
        "--age",
        type=int,
        default=None,
        help="Patient age"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch Streamlit web interface"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    # Parse arguments
    args = parser.parse_args()

    if args.web:
        logger.info("Launching Streamlit web interface...")
        import subprocess
        app_path = Path(__file__).parent / "app" / "main.py"
        subprocess.run([
            "streamlit",
            "run",
            str(app_path)
        ])
        return

    # If image provided, run analysis
    if args.image:
        # Initialize pipeline
        pipeline = LungCancerDetectionPipeline(config)

        # Prepare patient info
        patient_info = {
            'patient_id': args.patient_id,
            'name': args.name,
            'age': args.age,
            'study_date': datetime.now().strftime("%Y-%m-%d")
        }

        # Run analysis
        results = pipeline.analyze_image(
            args.image,
            patient_info=patient_info,
            generate_report=not args.no_report
        )

        if results and results.get('status') == 'success':
            # Display results
            print("\n" + "=" * 80)
            print("ANALYSIS RESULTS")
            print("=" * 80)

            risk = results['risk_assessment']
            print(f"\nRisk Assessment:")
            try:
                print(f"  Risk Score: {risk.risk_score:.3f}")
                print(f"  Risk Level: {risk.risk_level.value}")
                print(f"  Recommendation: {risk.recommendation}")
                print(f"  Follow-up: {risk.followup_period}")
            except Exception:
                print("  (risk details unavailable)")

            print(f"\nClinical Recommendations:")
            for i, rec in enumerate(results.get('recommendations', []), 1):
                try:
                    print(f"  {i}. {rec.action} (Priority: {rec.priority})")
                except Exception:
                    print(f"  {i}. {rec}")

            if results.get('report_path'):
                print(f"\nReport saved: {results['report_path']}")

            print("\n" + "=" * 80)
        else:
            print("Analysis failed. Check logs for details.")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nUse --web for web interface or --image <path> for image analysis")


if __name__ == "__main__":
    main()
