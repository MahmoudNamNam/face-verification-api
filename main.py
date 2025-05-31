from fastapi import FastAPI
from pydantic import BaseModel
import cv2
from deepface import DeepFace
import tempfile
import requests
import shutil
import logging
from typing import Optional
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define the request model
class FaceVerificationRequest(BaseModel):
    id_url: str
    ref_url: str

def download_image(url: str) -> Optional[str]:
    """Downloads an image from a URL and saves it to a temporary file."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        logger.info(f"Image downloaded successfully: {url}")
        return temp_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None

def detect_and_crop_face(image_path, detector_backend="mtcnn") -> Optional[str]:
    """Detects and crops the face from an image."""
    try:
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend=detector_backend, enforce_detection=False)
        if not faces:
            logger.warning("No faces detected.")
            return None
        
        face_info = faces[0]
        facial_area = face_info.get("facial_area", {})
        if not facial_area:
            logger.warning("No valid facial area found.")
            return None
        
        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
        image = cv2.imread(image_path)
        if image is None or w <= 0 or h <= 0:
            logger.error("Invalid face cropping dimensions.")
            return None
        
        cropped_face = image[y:y+h, x:x+w]
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(temp_path, cropped_face)
        logger.info(f"Face successfully cropped and saved at {temp_path}")
        return temp_path  
    except Exception as e:
        logger.error(f"Error during face detection: {e}")
        return None

@app.post("/verify")
async def verify_face(request: FaceVerificationRequest):
    """Verifies whether two faces belong to the same person."""
    try:
        id_path = download_image(request.id_url)
        ref_path = download_image(request.ref_url)
        
        if not id_path or not ref_path:
            return {"error": "Failed to download images."}
        
        cropped_face_path = detect_and_crop_face(id_path)
        if cropped_face_path:
            result = DeepFace.verify(
                img1_path=cropped_face_path, 
                img2_path=ref_path, 
                model_name="Facenet", 
                detector_backend="mtcnn"
            )
            threshold = 0.6  
            distance = result.get("distance", 1.0)
            is_match = distance < threshold
            
            logger.info(f"Face verification result: {result}")
            return {"match": is_match}
        else:
            return {"error": "Face detection failed for ID card image."}
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return {"error": str(e)}
    
@app.get("/")
async def root():
    return {"message": "Face Verification API!"}

# Run the FastAPI app with Uvicorn when executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
