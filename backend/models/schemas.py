from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class Role(str, Enum):
    DOCTOR = "doctor"
    ADMIN = "admin"
    USER = "user"

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    role: Role = Role.USER

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: str
    hashed_password: str

class UserResponse(UserBase):
    id: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[Role] = None

class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    recommendation: Optional[str] = None

class ConfidenceResult(BaseModel):
    prediction: str
    confidence: float

class PatientDetails(BaseModel):
    patient_id: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    smoker: Optional[bool] = None
    study_date: Optional[str] = None

class PatientHistoryCreate(BaseModel):
    patient_details: PatientDetails
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    recommendation: Optional[str] = None
    notes: Optional[str] = None

class PatientHistoryInDB(PatientHistoryCreate):
    id: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PatientHistoryResponse(PatientHistoryInDB):
    pass
