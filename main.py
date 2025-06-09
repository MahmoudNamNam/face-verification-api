import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import requests
import tempfile
import shutil
import logging
from typing import Optional

# --- ENVIRONMENT CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Setup ---
app = FastAPI()

# --- CORS for frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Schema ---
class FaceVerificationRequest(BaseModel):
    id_url: str
    ref_url: str

# --- Preload SFace model on startup ---
model_sface = None

@app.on_event("startup")
def preload_model():
    global model_sface
    model_sface = DeepFace.build_model("SFace")
    logger.info("✅ SFace model preloaded successfully.")

# --- Helper Function to Download Images ---
def download_image(url: str) -> Optional[str]:
    try:
        response = requests.get(url, stream=True, timeout=8)
        response.raise_for_status()
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        with open(temp_path, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return temp_path
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return None

# --- POST /verify ---
@app.post("/verify")
async def verify_face(request: FaceVerificationRequest):
    id_path = download_image(request.id_url)
    ref_path = download_image(request.ref_url)

    if not id_path or not ref_path:
        return {"error": "Failed to download one or both images."}

    try:
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=ref_path,
            model_name="SFace",
            model=model_sface,  # Reuse preloaded model
            detector_backend="opencv",
            enforce_detection=False
        )

        threshold = 0.6
        distance = result.get("distance", 1.0)
        is_match = distance < threshold

        os.remove(id_path)
        os.remove(ref_path)

        return {
            "match": is_match,
            "distance": round(distance, 4),
            "threshold": threshold
        }

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return {"error": "Verification error"}

# --- GET / ---
@app.get("/")
async def root():
    return {"message": "Face Verification API"}
