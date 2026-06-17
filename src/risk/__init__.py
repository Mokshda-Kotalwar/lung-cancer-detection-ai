"""
Risk Assessment and Recommendation Module
Comprehensive risk scoring and clinical recommendations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk classification"""
    LOW = "Low"
    INTERMEDIATE = "Intermediate"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class RiskAssessment:
    """Risk assessment result"""
    risk_score: float  # 0-1
    risk_level: RiskLevel
    confidence: float
    uncertainty: float
    key_features: List[str]
    contributing_factors: Dict[str, float]
    recommendation: str
    followup_period: str  # Days until follow-up


@dataclass
class ClinicalRecommendation:
    """Clinical recommendation"""
    action: str
    priority: str  # Low, Medium, High, Critical
    reasoning: str
    followup_days: int
    additional_tests: List[str]


class RiskScorer:
    """Risk scoring system"""
    
    def __init__(self, config):
        """Initialize risk scorer"""
        self.config = config
        self.risk_threshold_low = config.risk.risk_threshold_low
        self.risk_threshold_high = config.risk.risk_threshold_high
    
    def calculate_risk(self, classification_confidence: float,
                       detection_count: int,
                       detection_size: float,
                       detection_confidence: float,
                       clinical_features: Optional[Dict[str, float]] = None) -> RiskAssessment:
        """
        Calculate overall risk score
        
        Args:
            classification_confidence: Malignancy classification confidence (0-1)
            detection_count: Number of detected nodules
            detection_size: Size of largest nodule (mm)
            detection_confidence: Confidence of detection
            clinical_features: Additional clinical features
        
        Returns:
            RiskAssessment
        """
        # Base score from classification
        base_score = classification_confidence
        
        # Adjust for nodule characteristics
        size_factor = self._get_size_risk_factor(detection_size)
        count_factor = self._get_count_risk_factor(detection_count)
        
        # Combine factors (weighted average)
        risk_score = (
            base_score * 0.6 +
            size_factor * 0.2 +
            count_factor * 0.2
        )
        
        # Apply clinical features if provided
        if clinical_features and self.config.risk.use_clinical_features:
            clinical_score = self._calculate_clinical_score(clinical_features)
            risk_score = (
                risk_score * (1 - self.config.risk.clinical_feature_weight) +
                clinical_score * self.config.risk.clinical_feature_weight
            )
        
        # Clamp to 0-1
        risk_score = np.clip(risk_score, 0, 1)
        
        # Determine risk level
        risk_level = self._get_risk_level(risk_score)
        
        # Uncertainty quantification
        uncertainty = self._estimate_uncertainty(
            classification_confidence,
            detection_confidence
        )
        
        # Identify key features
        key_features = self._identify_key_features(
            classification_confidence,
            detection_count,
            detection_size
        )
        
        # Contributing factors
        contributing_factors = {
            "classification_confidence": classification_confidence,
            "size_factor": size_factor,
            "count_factor": count_factor,
            "uncertainty": uncertainty
        }
        
        recommendation = self._get_recommendation_text(risk_level)
        followup_period = self._get_followup_period(risk_level)
        
        return RiskAssessment(
            risk_score=float(risk_score),
            risk_level=risk_level,
            confidence=1 - uncertainty,
            uncertainty=float(uncertainty),
            key_features=key_features,
            contributing_factors=contributing_factors,
            recommendation=recommendation,
            followup_period=followup_period
        )
    
    def _get_size_risk_factor(self, size_mm: float) -> float:
        """Calculate risk factor based on nodule size"""
        if size_mm < 5:
            return 0.1
        elif size_mm < 10:
            return 0.3
        elif size_mm < 20:
            return 0.6
        elif size_mm < 30:
            return 0.8
        else:
            return 1.0
    
    def _get_count_risk_factor(self, count: int) -> float:
        """Calculate risk factor based on nodule count"""
        if count == 0:
            return 0.0
        elif count == 1:
            return 0.3
        elif count <= 3:
            return 0.6
        elif count <= 5:
            return 0.8
        else:
            return 1.0
    
    def _calculate_clinical_score(self, features: Dict[str, float]) -> float:
        """Calculate score from clinical features"""
        # Weighted combination of clinical features
        scores = {
            "smoking_pack_years": 0.3,
            "age": 0.2,
            "asbestos_exposure": 0.25,
            "family_history": 0.25
        }
        
        clinical_score = 0.0
        for feature, weight in scores.items():
            if feature in features:
                clinical_score += features[feature] * weight
        
        return np.clip(clinical_score, 0, 1)
    
    def _estimate_uncertainty(self, class_conf: float, det_conf: float) -> float:
        """Estimate model uncertainty"""
        if not self.config.risk.compute_uncertainty:
            return 0.0
        
        # Uncertainty inversely proportional to confidence
        uncertainty = (1 - class_conf) * (1 - det_conf)
        
        # Apply Monte Carlo dropout if available
        if self.config.risk.monte_carlo_iterations > 1:
            uncertainty *= np.sqrt(1 / self.config.risk.monte_carlo_iterations)
        
        return np.clip(uncertainty, 0, 1)
    
    def _identify_key_features(self, class_conf: float, count: int, size: float) -> List[str]:
        """Identify key features contributing to risk"""
        features = []
        
        if class_conf > 0.8:
            features.append("High malignancy confidence")
        elif class_conf > 0.5:
            features.append("Moderate malignancy risk")
        
        if count > 1:
            features.append(f"Multiple nodules ({count})")
        
        if size > 20:
            features.append(f"Large nodule ({size:.1f}mm)")
        elif size > 10:
            features.append(f"Medium nodule ({size:.1f}mm)")
        
        return features
    
    def _get_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from score"""
        if score < self.risk_threshold_low:
            return RiskLevel.LOW
        elif score < 0.5:
            return RiskLevel.INTERMEDIATE
        elif score < self.risk_threshold_high:
            return RiskLevel.INTERMEDIATE
        elif score < 0.9:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _get_recommendation_text(self, risk_level: RiskLevel) -> str:
        """Get recommendation text based on risk level"""
        recommendations = {
            RiskLevel.LOW: "Regular screening recommended. Annual follow-up.",
            RiskLevel.INTERMEDIATE: "Moderate risk detected. 3-6 month follow-up recommended.",
            RiskLevel.HIGH: "High risk detected. Urgent specialist consultation recommended.",
            RiskLevel.CRITICAL: "Critical risk. Immediate medical intervention required."
        }
        return recommendations.get(risk_level, "Medical consultation recommended")
    
    def _get_followup_period(self, risk_level: RiskLevel) -> str:
        """Get recommended follow-up period"""
        periods = {
            RiskLevel.LOW: "365 days",
            RiskLevel.INTERMEDIATE: "90 days",
            RiskLevel.HIGH: "30 days",
            RiskLevel.CRITICAL: "Immediate"
        }
        return periods.get(risk_level, "As advised by physician")


class RecommendationEngine:
    """Clinical recommendation generation"""
    
    def __init__(self, config):
        """Initialize recommendation engine"""
        self.config = config
        self.risk_scorer = RiskScorer(config)
    
    def generate_recommendations(self, risk_assessment: RiskAssessment,
                                 patient_info: Optional[Dict] = None) -> List[ClinicalRecommendation]:
        """
        Generate clinical recommendations
        
        Args:
            risk_assessment: Risk assessment result
            patient_info: Additional patient information
        
        Returns:
            List of clinical recommendations
        """
        recommendations = []
        
        # Base recommendations from risk level
        if risk_assessment.risk_level == RiskLevel.LOW:
            recommendations.append(ClinicalRecommendation(
                action="Annual CT screening",
                priority="Low",
                reasoning=f"Low risk score ({risk_assessment.risk_score:.2f}). Routine screening sufficient.",
                followup_days=365,
                additional_tests=[]
            ))
        
        elif risk_assessment.risk_level == RiskLevel.INTERMEDIATE:
            recommendations.append(ClinicalRecommendation(
                action="3-6 month follow-up CT",
                priority="Medium",
                reasoning=f"Intermediate risk ({risk_assessment.risk_score:.2f}). Close monitoring needed.",
                followup_days=90,
                additional_tests=["PET-CT", "Clinical assessment"]
            ))
        
        elif risk_assessment.risk_level == RiskLevel.HIGH:
            recommendations.append(ClinicalRecommendation(
                action="Urgent pulmonologist consultation",
                priority="High",
                reasoning=f"High risk ({risk_assessment.risk_score:.2f}). Specialist evaluation essential.",
                followup_days=14,
                additional_tests=["PET-CT", "Biopsy consideration", "Spirometry"]
            ))
            
            recommendations.append(ClinicalRecommendation(
                action="Repeat imaging in 1 month",
                priority="High",
                reasoning="Confirm findings and assess growth rate",
                followup_days=30,
                additional_tests=["CT scan"]
            ))
        
        else:  # CRITICAL
            recommendations.append(ClinicalRecommendation(
                action="URGENT: Same-day specialist referral",
                priority="Critical",
                reasoning=f"Critical risk ({risk_assessment.risk_score:.2f}). Immediate action required.",
                followup_days=0,
                additional_tests=["PET-CT", "Urgent biopsy", "Multidisciplinary tumor board"]
            ))
        
        # Add patient-specific recommendations
        if patient_info:
            recommendations.extend(self._get_patient_specific_recommendations(
                risk_assessment, patient_info
            ))
        
        return recommendations
    
    def _get_patient_specific_recommendations(self, risk_assessment: RiskAssessment,
                                              patient_info: Dict) -> List[ClinicalRecommendation]:
        """Generate patient-specific recommendations"""
        recommendations = []
        
        # Smoking cessation
        if patient_info.get("smoker", False):
            recommendations.append(ClinicalRecommendation(
                action="Smoking cessation program",
                priority="High",
                reasoning="Smoking increases lung cancer risk and progression",
                followup_days=0,
                additional_tests=["Nicotine replacement counseling"]
            ))
        
        # Age-specific recommendations
        age = patient_info.get("age", 0)
        if age > 75:
            recommendations.append(ClinicalRecommendation(
                action="Geriatric assessment before any intervention",
                priority="Medium",
                reasoning="Age-appropriate treatment planning essential",
                followup_days=14,
                additional_tests=["Cardiovascular assessment", "Pulmonary function tests"]
            ))
        
        return recommendations
    
    def get_next_steps(self, risk_assessment: RiskAssessment) -> List[str]:
        """Get recommended next steps"""
        steps = []
        
        if risk_assessment.risk_level == RiskLevel.LOW:
            steps.extend([
                "Schedule annual follow-up CT",
                "Maintain healthy lifestyle",
                "Avoid smoking"
            ])
        elif risk_assessment.risk_level == RiskLevel.INTERMEDIATE:
            steps.extend([
                "Schedule follow-up CT in 3 months",
                "Consult with pulmonologist",
                "Monitor for any symptoms"
            ])
        elif risk_assessment.risk_level == RiskLevel.HIGH:
            steps.extend([
                "URGENT: Contact pulmonologist",
                "Schedule PET-CT",
                "Prepare for possible biopsy",
                "Inform family of findings"
            ])
        else:  # CRITICAL
            steps.extend([
                "URGENT: Call pulmonologist immediately",
                "Go to emergency department if symptoms develop",
                "Have multidisciplinary assessment scheduled",
                "Consider clinical trial enrollment"
            ])
        
        return steps


class DifferentialDiagnosis:
    """Differential diagnosis suggestions"""
    
    @staticmethod
    def get_differential_diagnoses(nodule_characteristics: Dict) -> List[Dict]:
        """
        Get differential diagnosis suggestions based on nodule characteristics
        
        Args:
            nodule_characteristics: Dict with size, shape, density, location, etc.
        
        Returns:
            List of possible diagnoses with probabilities
        """
        diagnoses = []
        
        size = nodule_characteristics.get("size", 0)
        
        # Rule-based differential diagnosis
        if size < 5:
            diagnoses.extend([
                {"condition": "Granuloma", "probability": 0.4},
                {"condition": "Infectious lesion", "probability": 0.3},
                {"condition": "Ground-glass nodule", "probability": 0.2},
            ])
        elif size < 15:
            diagnoses.extend([
                {"condition": "Adenocarcinoma", "probability": 0.3},
                {"condition": "Squamous cell carcinoma", "probability": 0.25},
                {"condition": "Granuloma", "probability": 0.2},
            ])
        else:
            diagnoses.extend([
                {"condition": "Lung cancer (high probability)", "probability": 0.6},
                {"condition": "Metastatic lesion", "probability": 0.2},
                {"condition": "Infection/Abscess", "probability": 0.1},
            ])
        
        return diagnoses


if __name__ == "__main__":
    print("Risk assessment and recommendation module loaded successfully")
