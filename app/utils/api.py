import requests
import json
import io
from PIL import Image

BACKEND_URL = "http://127.0.0.1:8000"

def check_backend_health():
    """Checks if the backend is reachable."""
    try:
        response = requests.get(f"{BACKEND_URL}/health/", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def authenticate_user(email, password):
    """Authenticates user and returns JWT token."""
    try:
        data = {"username": email, "password": password}
        response = requests.post(f"{BACKEND_URL}/auth/login", data=data, timeout=5)
        if response.status_code == 200:
            return response.json().get("access_token")
    except Exception:
        pass
    return None

def get_patient_history(token):
    """Fetches patient prediction history."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BACKEND_URL}/history/", headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

def run_prediction(token, file_bytes, filename, patient_data):
    """Runs the AI prediction pipeline."""
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (filename, file_bytes, "image/png")}
    data = {"patient_details_json": json.dumps(patient_data)}
    
    response = requests.post(f"{BACKEND_URL}/predict/", headers=headers, files=files, data=data)
    return response

def get_gradcam(token, file_bytes, filename):
    """Fetches the Grad-CAM image for a given scan."""
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (filename, file_bytes, "image/png")}
    
    response = requests.post(f"{BACKEND_URL}/predict/gradcam", headers=headers, files=files)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None

def generate_report(token, file_bytes, filename, patient_data):
    """Generates a PDF report."""
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (filename, file_bytes, "image/png")}
    data = {"patient_details_json": json.dumps(patient_data)}
    
    response = requests.post(f"{BACKEND_URL}/predict/report", headers=headers, files=files, data=data)
    return response
