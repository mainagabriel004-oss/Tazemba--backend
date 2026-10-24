from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import PIL.Image
import io

app = FastAPI(title="Tazemba AI Engine")

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Tazemba AI Engine is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image bytes
    image_bytes = await file.read()
    image = PIL.Image.open(io.BytesIO(image_bytes))
    
    # Run inference with confidence threshold at 40%
    results = model(image, conf=0.40)
    
    detections = []
    class_counts = {}
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Aggregate counts per detected class
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "box": [round(x, 2) for x in box.xyxy[0].tolist()]
            })

    return {
        "filename": file.filename,
        "summary": {
            "total_objects": len(detections),
            "counts": class_counts
        },
        "detections": detections
    }
