from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

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

class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]

class PatientHistoryCreate(BaseModel):
    patient_id: Optional[str] = None
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    notes: Optional[str] = None

class PatientHistoryInDB(PatientHistoryCreate):
    id: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PatientHistoryResponse(PatientHistoryInDB):
    pass
