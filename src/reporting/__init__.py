"""
Report Generation Module using ReportLab
Creates professional PDF reports with AI findings and recommendations
"""

import io
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate professional medical reports"""
    
    def __init__(self, config):
        """Initialize report generator"""
        self.config = config
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
            
            self.reportlab = {
                'SimpleDocTemplate': SimpleDocTemplate,
                'Paragraph': Paragraph,
                'Spacer': Spacer,
                'Table': Table,
                'TableStyle': TableStyle,
                'PageBreak': PageBreak,
                'Image': Image,
                'colors': colors,
                'inch': inch,
                'cm': cm,
                'TA_JUSTIFY': TA_JUSTIFY,
                'TA_CENTER': TA_CENTER,
                'TA_LEFT': TA_LEFT,
                'getSampleStyleSheet': getSampleStyleSheet,
                'ParagraphStyle': ParagraphStyle,
                'A4': A4,
                'letter': letter
            }
            logger.info("ReportLab loaded successfully")
        except ImportError:
            logger.error("reportlab not installed. Install: pip install reportlab")
            self.reportlab = None
    
    def generate_pdf_report(self, output_path: str, patient_info: Dict,
                           detection_result, classification_result,
                           risk_assessment, recommendations: List,
                           gradcam_image: Optional[np.ndarray] = None,
                           original_image: Optional[np.ndarray] = None) -> bool:
        """
        Generate comprehensive PDF report
        
        Args:
            output_path: Path to save PDF
            patient_info: Patient information dict
            detection_result: Detection results
            classification_result: Classification results
            risk_assessment: Risk assessment results
            recommendations: Clinical recommendations
            gradcam_image: Grad-CAM visualization
            original_image: Original medical image
        
        Returns:
            True if successful
        """
        if self.reportlab is None:
            logger.error("Cannot generate report without reportlab")
            return False
        
        try:
            # Create PDF
            buffer = io.BytesIO()
            
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
            from reportlab.lib import colors
            
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2e5c96'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            # Title
            story.append(Paragraph("Lung Cancer Detection Report", title_style))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))
            
            # Patient Information Section
            story.append(Paragraph("Patient Information", heading_style))
            patient_data = [
                ["Field", "Value"],
                ["Patient ID", str(patient_info.get("patient_id", "N/A"))],
                ["Name", str(patient_info.get("name", "N/A"))],
                ["Age", str(patient_info.get("age", "N/A"))],
                ["Gender", str(patient_info.get("gender", "N/A"))],
                ["Study Date", str(patient_info.get("study_date", datetime.now().strftime("%Y-%m-%d")))],
            ]
            
            patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c96')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 0.3 * inch))
            
            # Findings Section
            story.append(Paragraph("AI Analysis Findings", heading_style))
            
            findings_text = f"""
            <b>Detection Results:</b><br/>
            Number of nodules detected: {len(detection_result.boxes) if hasattr(detection_result, 'boxes') else 0}<br/>
            Detection confidence: {(detection_result.confidences.mean() * 100 if hasattr(detection_result, 'confidences') and len(detection_result.confidences) > 0 else 0):.1f}%<br/>
            <br/>
            <b>Classification Results:</b><br/>
            Predicted class: {classification_result.class_name if classification_result else 'N/A'}<br/>
            Confidence: {(classification_result.confidence * 100 if classification_result else 0):.1f}%<br/>
            """
            
            story.append(Paragraph(findings_text, styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))
            
            # Risk Assessment Section
            story.append(Paragraph("Risk Assessment", heading_style))
            
            risk_text = f"""
            <b>Overall Risk Score:</b> {risk_assessment.risk_score:.2f}/1.0<br/>
            <b>Risk Level:</b> <font color="{self._get_risk_color(risk_assessment.risk_level)}">{risk_assessment.risk_level.value}</font><br/>
            <b>Confidence:</b> {(risk_assessment.confidence * 100):.1f}%<br/>
            <b>Model Uncertainty:</b> {(risk_assessment.uncertainty * 100):.1f}%<br/>
            <br/>
            <b>Key Features:</b><br/>
            """
            
            for feature in risk_assessment.key_features:
                risk_text += f"• {feature}<br/>"
            
            story.append(Paragraph(risk_text, styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))
            
            # Recommendations Section
            story.append(Paragraph("Clinical Recommendations", heading_style))
            
            for i, rec in enumerate(recommendations, 1):
                rec_text = f"""
                <b>Recommendation {i}:</b> {rec.action}<br/>
                Priority: <font color="{self._get_priority_color(rec.priority)}">{rec.priority}</font><br/>
                Reasoning: {rec.reasoning}<br/>
                Follow-up: {rec.followup_days} days<br/>
                """
                if rec.additional_tests:
                    rec_text += f"Additional Tests: {', '.join(rec.additional_tests)}<br/>"
                rec_text += "<br/>"
                
                story.append(Paragraph(rec_text, styles['Normal']))
            
            story.append(Spacer(1, 0.3 * inch))
            
            # Visualizations
            if original_image is not None or gradcam_image is not None:
                story.append(PageBreak())
                story.append(Paragraph("AI Explainability - Grad-CAM Visualization", heading_style))
                
                # Save images temporarily
                if gradcam_image is not None:
                    import cv2
                    gradcam_path = Path(output_path).parent / "gradcam_temp.png"
                    cv2.imwrite(str(gradcam_path), gradcam_image)
                    story.append(Image(str(gradcam_path), width=5*inch, height=5*inch))
                    story.append(Paragraph(
                        "Grad-CAM shows the regions of interest contributing most to the AI prediction.",
                        styles['Normal']
                    ))
            
            # Disclaimer
            story.append(Spacer(1, 0.3 * inch))
            story.append(Paragraph("Disclaimer", heading_style))
            
            disclaimer_text = """
            This report is generated by an AI system for supporting clinical decision-making. 
            It should not be used as a standalone diagnostic tool. 
            All findings should be reviewed and validated by qualified medical professionals.
            The AI predictions are based on the analyzed images and available clinical data.
            Final diagnosis and treatment decisions must be made by healthcare providers.
            """
            
            story.append(Paragraph(disclaimer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return False
    
    def generate_html_report(self, output_path: str, patient_info: Dict,
                            detection_result, classification_result,
                            risk_assessment, recommendations: List) -> bool:
        """Generate HTML report"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Lung Cancer Detection Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #1f4788; }}
                    h2 {{ color: #2e5c96; border-bottom: 2px solid #2e5c96; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #2e5c96; color: white; }}
                    .risk-low {{ color: green; }}
                    .risk-intermediate {{ color: orange; }}
                    .risk-high {{ color: red; }}
                    .risk-critical {{ color: darkred; font-weight: bold; }}
                    .disclaimer {{ background-color: #ffe6e6; padding: 10px; border-left: 4px solid red; }}
                </style>
            </head>
            <body>
                <h1>Lung Cancer Detection Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Patient Information</h2>
                <table>
                    <tr><th>Field</th><th>Value</th></tr>
                    <tr><td>Patient ID</td><td>{patient_info.get('patient_id', 'N/A')}</td></tr>
                    <tr><td>Name</td><td>{patient_info.get('name', 'N/A')}</td></tr>
                    <tr><td>Age</td><td>{patient_info.get('age', 'N/A')}</td></tr>
                    <tr><td>Gender</td><td>{patient_info.get('gender', 'N/A')}</td></tr>
                </table>
                
                <h2>AI Analysis Findings</h2>
                <p>Nodules detected: {len(detection_result.boxes) if hasattr(detection_result, 'boxes') else 0}</p>
                <p>Classification: {classification_result.class_name if classification_result else 'N/A'}</p>
                <p>Confidence: {(classification_result.confidence * 100):.1f}%</p>
                
                <h2>Risk Assessment</h2>
                <p class="risk-{self._get_risk_class(risk_assessment.risk_level)}">
                    <strong>Risk Score:</strong> {risk_assessment.risk_score:.2f}<br/>
                    <strong>Risk Level:</strong> {risk_assessment.risk_level.value}<br/>
                    <strong>Confidence:</strong> {(risk_assessment.confidence * 100):.1f}%
                </p>
                
                <h2>Clinical Recommendations</h2>
            """
            
            for i, rec in enumerate(recommendations, 1):
                html_content += f"""
                <h3>Recommendation {i}</h3>
                <p><strong>Action:</strong> {rec.action}</p>
                <p><strong>Priority:</strong> {rec.priority}</p>
                <p><strong>Reasoning:</strong> {rec.reasoning}</p>
                <p><strong>Follow-up:</strong> {rec.followup_days} days</p>
                """
            
            html_content += """
                <div class="disclaimer">
                    <h3>Disclaimer</h3>
                    <p>This report is generated by an AI system. All findings must be validated by qualified medical professionals.</p>
                </div>
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return False
    
    @staticmethod
    def _get_risk_color(risk_level) -> str:
        """Get color for risk level"""
        colors_map = {
            "Low": "green",
            "Intermediate": "orange",
            "High": "red",
            "Critical": "darkred"
        }
        return colors_map.get(risk_level.value, "black")
    
    @staticmethod
    def _get_risk_class(risk_level) -> str:
        """Get CSS class for risk level"""
        class_map = {
            "Low": "low",
            "Intermediate": "intermediate",
            "High": "high",
            "Critical": "critical"
        }
        return class_map.get(risk_level.value, "low")
    
    @staticmethod
    def _get_priority_color(priority: str) -> str:
        """Get color for priority"""
        colors_map = {
            "Low": "green",
            "Medium": "orange",
            "High": "red",
            "Critical": "darkred"
        }
        return colors_map.get(priority, "black")


if __name__ == "__main__":
    print("Report generation module loaded successfully")
