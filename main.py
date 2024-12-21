from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from PIL import Image
import cv2
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLOv5 model from the specified path
model_path = "A:\\BeeMonitor\\bee_detection\\yolov5\\runs\\train\\exp49\\weights\\best.pt"
try:
    logger.info("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Initialize FastAPI app
app = FastAPI(title="BeeMonitor API",
             description="API for detecting wasps in video footage",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],  # Enable GET and POST methods
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the BeeMonitor API!",
        "status": "running",
        "endpoints": {
            "detect": "/detect (GET) - Starts detection on a video file"
        }
    }

@app.get("/detect")  # Changed to GET method for simplicity
async def detect():
    try:
        logger.info("Starting detection process")

        # Path to the pre-uploaded video in the backend
        video_path = "C:\\Users\\asus\\Downloads\\wasptest.mp4"  # Change the video path manually
        
        # Verify video file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found at {video_path}")
            raise HTTPException(status_code=404, detail="Video file not found")
        
        logger.info(f"Opening video from {video_path}")
        
        # Open the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error("Failed to open video file")
            raise HTTPException(status_code=500, detail="Failed to open video file")
        
        wasps_detected = False
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit when no more frames are available
            
            frame_count += 1
            if frame_count % 10 == 0:  # Log every 10th frame
                logger.info(f"Processing frame {frame_count}")

            # Convert the frame to RGB format (PIL image format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Perform detection on the frame
            logger.info(f"Performing detection on frame {frame_count}")
            results = model(pil_image)
            detections = results.pandas().xyxy[0]

            logger.info(f"Frame {frame_count} detections: {detections[['name', 'confidence']]}")
            
            # Check for wasps in the detections
            if any(name.lower() == 'wasp' or name.lower() == 'not bee' for name in detections['name']):
                logger.info("Wasp detected in video")
                wasps_detected = True
                break  # Exit early once a wasp is detected

        cap.release()  # Release the video capture object
        logger.info(f"Video processing completed. Processed {frame_count} frames")

        # Return the result
        if wasps_detected:
            return {
                "detection": "Wasp detected in video"
            }
        else:
            return {
                "detection": "No wasps detected in video"
            }

    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing the video: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
