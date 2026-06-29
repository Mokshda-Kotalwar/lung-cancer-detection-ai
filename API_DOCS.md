# FastAPI Endpoints Documentation

This document describes the API endpoints exposed by the Lung Cancer Detection backend.

## Base URL
`http://localhost:8000` (or the equivalent Render External URL)

## Authentication
All endpoints except `/auth/signup`, `/auth/login`, and `/health` require a JWT Token.
Pass the token in the headers as:
`Authorization: Bearer <your_token>`

---

## 1. Authentication Endpoints (`/auth`)

### `POST /auth/signup`
Creates a new user account with a designated role (Admin, Doctor, User).
- **Body:** JSON `{"email": "...", "password": "...", "full_name": "...", "role": "Doctor"}`
- **Response:** JSON representation of the created user.

### `POST /auth/login`
Authenticates a user and returns a JWT access token.
- **Body:** Form Data `username=<email>&password=<password>`
- **Response:** `{"access_token": "...", "token_type": "bearer"}`

---

## 2. Diagnostics Endpoints (`/predict`)

### `POST /predict/`
Analyzes an uploaded medical image and returns classification and risk scoring data.
- **Form Data:** 
  - `file`: The medical image file (PNG/JPG/DICOM).
  - `patient_details_json`: Stringified JSON containing patient metadata (e.g. `{"age": 45, "smoker": false}`).
- **Response (200 OK):**
```json
{
  "prediction": "Malignant",
  "confidence": 0.92,
  "probabilities": {"Benign": 0.05, "Malignant": 0.92, "Uncertain": 0.03},
  "risk_score": 85.5,
  "risk_level": "High",
  "recommendation": "Immediate biopsy recommended."
}
```

### `POST /predict/gradcam`
Generates a visual explainability heatmap overlay over the original image.
- **Form Data:** 
  - `file`: The medical image file.
- **Response (200 OK):** Binary PNG image file (can be rendered directly in `<img>` tags).

---

## 3. History Endpoints (`/history`)

### `GET /history/`
Retrieves the logged diagnostic history for the currently authenticated user.
- **Query Parameters:**
  - `limit`: Integer (default 20, max 100).
  - `search_query`: Optional string to filter by patient ID or Name.
- **Response (200 OK):** Array of historical diagnostic JSON objects.

---

## 4. System Endpoints (`/health`)

### `GET /health/`
Simple health check for deployment readiness probes (Docker/Render).
- **Response (200 OK):** `{"status": "ok", "service": "lung-cancer-detection-api"}`
