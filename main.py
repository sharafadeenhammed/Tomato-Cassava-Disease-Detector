from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import time

app = FastAPI(title="Crop Disease Classification Server")

def predict_crop_disease(image: Image.Image) -> dict:
    """
    Placeholder function for your Cassava / Tomato Disease Detection model.
    Replace this with your trained PyTorch, TensorFlow, or ONNX inference model.
    """
    # Example logic: image width/height verification or inference
    width, height = image.size
    
    # Simulate inference results
    # Sample predictions: "Cassava Mosaic Disease", "Tomato Early Blight", "Healthy Leaf", etc.
    detected_class = "Tomato Early Blight"
    confidence = 0.94

    return {
        "status": "success",
        "crop": "Tomato",
        "disease": detected_class,
        "confidence": float(confidence)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes from incoming request
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run prediction model
        result = predict_crop_disease(image)
        
        print(f"[{time.strftime('%H:%M:%S')}] Detected: {result['disease']} ({result['confidence']*100:.1f}%)")
        return result

    except Exception as e:
        print(f"Error processing image: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Run server accessible on local network (0.0.0.0) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)