

# --- DETECTED CLASSES ---
# Class [0]: Cassava_Cassava_Bacterial_Blight
# Class [1]: Cassava_Cassava_Brown_Streak_Disease
# Class [2]: Cassava_Cassava_Green_Mottle
# Class [3]: Cassava_Cassava_Mosaic_Disease
# Class [4]: Cassava_Healthy
# Class [5]: Tomato_Bacterial_spot
# Class [6]: Tomato_Early_blight
# Class [7]: Tomato_Late_blight
# Class [8]: Tomato_Leaf_Mold
# Class [9]: Tomato_Septoria_leaf_spot
# Class [10]: Tomato_Spider_mites_Two_spotted_spider_mite
# Class [11]: Tomato__Target_Spot
# Class [12]: Tomato__Tomato_YellowLeaf__Curl_Virus
# Class [13]: Tomato__Tomato_mosaic_virus
# Class [14]: Tomato_healthy

import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import onnxruntime as ort

app = FastAPI(title="Crop Disease Detection API")
# CONFIDENCE_THRESHOLD = 0.70  # Require at least 70% confidence
CONFIDENCE_THRESHOLD = 0.55  # Require at least 70% confidence
MIN_GREEN_RATIO = 0.10       # Require at least 10% green/plant pixels in the image

# 1. Load the ONNX model
session = ort.InferenceSession("crop_disease_mobilenet.onnx")
input_name = session.get_inputs()[0].name

# 2. Define the exact class mapping from your training output
CLASS_NAMES = [
    "Cassava_Cassava_Bacterial_Blight",
    "Cassava_Cassava_Brown_Streak_Disease",
    "Cassava_Cassava_Green_Mottle",
    "Cassava_Cassava_Mosaic_Disease",
    "Cassava_Healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


def is_plant_present(image_bytes: bytes, min_ratio: float = MIN_GREEN_RATIO) -> bool:
    """
    Checks if the raw image contains enough green/plant-like HSV color pixels
    to prevent false positives on bare walls, floors, or background noise.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if cv_img is None:
        return False

    # Convert BGR image to HSV color space
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    
    # Range covering typical plant foliage hues (adjust if dealing with yellowed leaves)
    lower_green = np.array([20, 30, 30])
    upper_green = np.array([85, 255, 255])
    
    # Filter for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (cv_img.shape[0] * cv_img.shape[1])
    print("green ratio: ", green_ratio)
    
    return green_ratio >= 0.5


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocesses raw image bytes from ESP32-CAM to match PyTorch training transforms."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    
    # Convert to float array and normalize [0, 1]
    img_data = np.array(image, dtype=np.float32) / 255.0
    
    # Standard ImageNet Normalization used during training: (x - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # Change shape from (H, W, C) to (C, H, W) and add Batch dimension -> (1, C, H, W)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data


@app.get("/")
async def status():
    return {
        'error': False,
        'status':'server running'
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # 1. First Guardrail: Ensure a plant is actually in the frame
    if not is_plant_present(image_bytes):
        print("Rejection: No plant/leaf color detected in image.")
        return {
            "status": "no_plant_detected",
            "class_id": -1,
            "disease": "Unknown / Background Wall",
            "confidence": 0.0
        }

    # 2. Preprocess and run ONNX inference
    input_tensor = preprocess_image(image_bytes)
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    
    # Apply softmax to get confidence percentages
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    
    # 3. Second Guardrail: Enforce Confidence Floor
    print("confidence: ", confidence)
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"Low Confidence: {confidence*100:.2f}% on class {CLASS_NAMES[predicted_idx]}")
        return {
            "status": "low_confidence",
            "class_id": -1,
            "disease": "Uncertain / Unclear Image",
            "confidence": round(confidence * 100, 2)
        }

    # 4. Success Response
    print(f"Disease Detected: {CLASS_NAMES[predicted_idx]} ({confidence*100:.2f}%)")
    return {
        "status": "success",
        "class_id": predicted_idx,
        "disease": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence * 100, 2)
    }

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)