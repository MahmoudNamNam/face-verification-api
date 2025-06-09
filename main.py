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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class FaceVerificationRequest(BaseModel):
    id_url: str
    ref_url: str

def download_image(url: str) -> Optional[str]:
    """Download an image from a URL."""
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

@app.post("/verify")
async def verify_face(request: FaceVerificationRequest):
    """Verify if two faces match."""
    id_path = download_image(request.id_url)
    ref_path = download_image(request.ref_url)

    if not id_path or not ref_path:
        return {"error": "Failed to download one or both images."}

    try:
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=ref_path,
            model_name="SFace",  
            detector_backend="opencv",  
            enforce_detection=False
        )

        threshold = 0.6
        distance = result.get("distance", 1.0)
        is_match = distance < threshold

        # Clean up temp files
        os.remove(id_path)
        os.remove(ref_path)

        return {
            "match": is_match,
        }

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return {"error": "Verification error"}

@app.get("/")
async def root():
    return {"message": "Face Verification API"}
