from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import requests
import cv2
import numpy as np
import tempfile
import logging
import os

# --- ENVIRONMENT CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Init ---
app = FastAPI()
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

# --- Image Downloader ---
def download_image_to_tempfile(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logger.error(f"Image download failed from {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Image download failed: {e}")

# --- Face Cropper ---
def detect_and_crop_face(image_path: str, backend="mtcnn") -> str:
    try:
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend=backend, enforce_detection=False)
        if not faces:
            raise ValueError("No face detected.")

        face = faces[0]["facial_area"]
        img = cv2.imread(image_path)
        cropped = img[face["y"]:face["y"]+face["h"], face["x"]:face["x"]+face["w"]]

        temp_cropped = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_cropped.name, cropped)
        return temp_cropped.name
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=422, detail=f"Face detection failed: {e}")

# --- API Endpoint ---
@app.post("/verify")
async def verify_face(request: FaceVerificationRequest):
    try:
        id_img = download_image_to_tempfile(request.id_url)
        ref_img = download_image_to_tempfile(request.ref_url)
        cropped_id_img = detect_and_crop_face(id_img)

        result = DeepFace.verify(
            img1_path=cropped_id_img,
            img2_path=ref_img,
            model_name="Facenet",
            detector_backend="mtcnn"
        )

        threshold = 0.6
        distance = result.get("distance", 1.0)
        is_match = distance < threshold

        return {"match": is_match}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return {"error": str(e)}

# --- Health Check ---
@app.get("/")
def root():
    return {"message": "Face Verification API is running."}
