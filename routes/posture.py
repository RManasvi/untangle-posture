from fastapi import APIRouter, File, UploadFile
import cv2
import numpy as np
from posture_analyzer import PostureAnalyzer

router = APIRouter()
analyzer = PostureAnalyzer()

@router.post("/analyze_posture")
async def analyze_posture(image: UploadFile = File(...)):
    """Receives a frame, analyzes posture, and returns metrics via API."""
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid image"}
            
        analyzer.analyze_frame(frame)
        return analyzer.get_results()
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/latest_posture")
async def get_latest_posture():
    """Returns the results of the last posture analysis."""
    return analyzer.get_results()
